import json
import os
import sys
from dataclasses import replace
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = PROJECT_ROOT / "app"
EVAL_CONFIG_PATH = PROJECT_ROOT / "eval" / "eval_config.json"
EVAL_TRACE_PATH = PROJECT_ROOT / "eval" / "eval_traces.jsonl"

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from agent import (
    ask_course_agent,
    detect_task_type,
    merge_retrieval_results,
    narrow_summary_results,
)
from config import get_settings
from loader import load_document_chunks, load_documents
from retriever import retrieve_chunks, retrieve_documents
from reranker import build_reranker
from vector_index_service import build_vector_store_with_cache

#   {
#     "id": "qa-tool-use-001", //id唯一标识，后面跑评估脚本时很好用
#     "task_type": "qa", //问题类型，目前设计了三种：qa、summary、study_plan
#     "question": "tool use 是什么？", //就是实际要问系统的问题。
#     "expected_primary_source_contains": [ // 我期望首条来源至少包含这些路径片段之一。
#       "2-3-ai-agents-for-beginners/04-tool-use/notebook-summary.md"
#     ],
#     "expected_source_group": "2-3-ai-agents-for-beginners", //表示我期望结果主要来自哪个课程簇。
#     "allow_cross_document": true, // 对 summary为false，对 qa、study_plan通常可以是：true,表示是否允许结果来源跨文档（同一课程簇内）。
#     "notes": "首条应命中 04 Tool Use，允许后续补充来自同课程组的 Agent 相关模块。" //给你自己看的，不参与程序逻辑也没关系。后面分析失败案例时很有用。
#   },
def load_questions() -> list[dict]:
    questions_path = PROJECT_ROOT / "eval" / "questions.json"
    with questions_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_eval_config() -> dict:
    with EVAL_CONFIG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)

def strip_chunk_id(source: str) -> str:
    return source.split("#", 1)[0]


def get_course_group(source: str) -> str | None:
    parts = Path(source).parts
    for part in parts:
        if part.startswith(("1-", "2-")):
            return part
    return None

def build_active_reranker(settings):
    if settings.retrieval.reranker_provider == "none":
        return None

    return build_reranker(
        settings.retrieval.reranker_provider,
        settings.retrieval.reranker_model_name,
        settings.retrieval.reranker_cache_dir,
    )

def should_run_agent_eval() -> bool:
    """读取 RUN_AGENT_EVAL 开关，控制是否执行真实模型调用。"""
    config = load_eval_config()
    default_value = str(config.get("default_run_agent_eval", False)).lower()
    raw_value = os.getenv("RUN_AGENT_EVAL", default_value).strip().lower()
    return raw_value not in {"0", "false", "no", "off"}

def get_eval_modes() -> list[str]:
    """读取 EVAL_MODES 开关， 控制本次评估运行哪些检索模式。"""
    config = load_eval_config()
    default_modes = config.get("default_modes", ["chunk", "vector", "hybrid"])
    default_value = ",".join(default_modes)
    raw_value = os.getenv("EVAL_MODES", default_value).strip().lower()
    modes = [item.strip() for item in raw_value.split(",") if item.strip()]
    allowed_modes = {"document", "chunk", "vector", "hybrid"}

    if not modes:
        raise ValueError("EVAL_MODES must contain at least one mode.")
    
    invalid_modes = [mode for mode in modes if mode not in allowed_modes]
    if invalid_modes:
        raise ValueError("EVAL_MODES contains unsupported modes: " + ", ".join(invalid_modes))
    
    return modes


def get_eval_tags() -> set[str] | None:
    """读取 EVAL_TAGS 开关，控制本次评估只运行哪些标签样本。"""
    config = load_eval_config()
    default_tags = config.get("default_tags", [])
    default_value = ",".join(default_tags)
    raw_value = os.getenv("EVAL_TAGS", default_value).strip().lower()
    if not raw_value:
        return None

    tags = {item.strip() for item in raw_value.split(",") if item.strip()}
    return tags or None

def should_write_eval_traces() -> bool:
    """读取 WRITE_EVAL_TRACES 开关，控制是否把评估过程中的检索 trace 写入到文件。"""
    raw_value = os.getenv("WRITE_EVAL_TRACES", "true").strip().lower()
    return raw_value not in {"0", "false", "no", "off"}


def retrieve_for_mode(
    question: str,
    mode: str,
    settings,
    documents,
    chunks,
    vector_store,
    memory: dict | None = None,
    reranker=None,
):
    if mode == "document":
        retrieved = retrieve_documents(
            query=question,
            documents=documents,
            top_k=settings.retrieval.retrieval_top_k,
        )
    elif mode == "chunk":
        retrieved = retrieve_chunks(
            query=question,
            chunks=chunks,
            top_k=settings.retrieval.retrieval_top_k,
        )
    elif mode == "vector":
        if vector_store is None:
            raise ValueError("vector mode requires vector_store")
        retrieved = vector_store.search(
            query=question,
            top_k=settings.retrieval.retrieval_top_k,
        )
    elif mode == "hybrid":
        if vector_store is None:
            raise ValueError("hybrid mode requires vector_store")

        candidate_top_k = max(
            settings.retrieval.retrieval_top_k * settings.retrieval.hybrid_candidate_multiplier,
            settings.retrieval.hybrid_candidate_minimum,
        )

        lexical_results = retrieve_chunks(
            query=question,
            chunks=chunks,
            top_k=candidate_top_k,
        )
        vector_results = vector_store.search(
            query=question,
            top_k=candidate_top_k,
        )
        retrieved = merge_retrieval_results(
            lexical_results,
            vector_results,
            top_k=settings.retrieval.retrieval_top_k,
        )
        if reranker is not None:
            retrieved = reranker.rerank(
                question,
                retrieved,
                top_k=settings.retrieval.retrieval_top_k,
            )
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    task_type = detect_task_type(question)
    if task_type == "summary":
        retrieved = narrow_summary_results(
            retrieved,
            strategy=settings.retrieval.summary_strategy,
        )

    return retrieved


def evaluate_sources(sample: dict, sources: list[str]) -> dict:
    """评估检索结果的来源是否符合预期，包括首条来源、来源的课程组覆盖情况，以及 summary 任务是否被跨文档污染等。"""
    first_source = sources[0] if sources else ""
    first_source_clean = strip_chunk_id(first_source) if first_source else ""

    expected_primary = sample.get("expected_primary_source_contains", [])
    primary_hit = any(part in first_source_clean for part in expected_primary)

    expected_group = sample.get("expected_source_group")
    source_groups = [get_course_group(strip_chunk_id(source)) for source in sources]
    first_group = source_groups[0] if source_groups else None
    group_hit = first_group == expected_group

    expected_any_groups = sample.get("expected_any_source_groups", [])
    actual_groups = {group for group in source_groups if group}
    covered_groups = [
        group for group in expected_any_groups
        if group in actual_groups
    ]
    group_coverage = (
        len(covered_groups) / len(expected_any_groups)
        if expected_any_groups else None
    )

    allow_cross_document = sample.get("allow_cross_document", True)
    source_docs = {strip_chunk_id(source) for source in sources}
    summary_polluted = (
        sample.get("task_type") == "summary"
        and not allow_cross_document
        and len(source_docs) > 1
    )

    return {
        "first_source": first_source_clean,
        "primary_hit": primary_hit,
        "first_group": first_group,
        "group_hit": group_hit,
        "summary_polluted": summary_polluted,
        "source_count": len(source_docs),
        "sources": sources,
        "expected_any_source_groups": expected_any_groups,
        "covered_groups": covered_groups,
        "group_coverage": group_coverage,
    }

def is_mode_enabled_for_sample(sample: dict, mode: str) -> bool:
    """判断当前样本是否允许在指定mode下参与评估。"""
    enabled_modes = sample.get("enabled_modes")
    if not enabled_modes:
        return True
    return mode in enabled_modes


def is_tag_enabled_for_sample(sample: dict, active_tags: set[str] | None) -> bool:
    """判断当前样本是否命中本次评估要求的标签。"""
    if active_tags is None:
        return True

    sample_tags = {item.strip().lower() for item in sample.get("tags", []) if item.strip()}
    return bool(sample_tags & active_tags)

def evaluate_sample(sample: dict, retrieved: list) -> dict:
    evaluated = evaluate_sources(
        sample,
        [item.source for item in retrieved],
    )
    evaluated.update(
        {
            "id": sample["id"],
            "task_type": sample["task_type"],
            "question": sample["question"],
        }
    )
    return evaluated

def evaluate_answer_content(sample: dict, answer_text: str) -> dict:
    """评估答案文本是否包含期望内容，以及是否包含不该出现的内容"""
    expected_answer_contains = sample.get("expected_answer_contains", [])
    forbidden_answer_contains = sample.get("forbidden_answer_contains", [])

    expected_hits = [item for item in expected_answer_contains if item in answer_text] # 这个列表包含了所有在答案文本中命中的 expected_answer_contains 项。理论上，如果答案质量很好，并且 sample 设计合理，这个列表应该包含所有的 expected_answer_contains 项，也就是 expected_hits 的长度应该等于 expected_answer_contains 的长度。
    forbidden_hits = [item for item in forbidden_answer_contains if item in answer_text] # 这个列表包含了所有在答案文本中命中的 forbidden_answer_contains 项。理论上，如果答案质量很好，并且 sample 设计合理，这个列表应该是空的，也就是 forbidden_hits 的长度应该为 0。

    expected_answer_hit = (len(expected_hits) == len(expected_answer_contains) if expected_answer_contains else None) # 如果有 expected_answer_contains，就要求全部命中才算 hit；如果没有 expected_answer_contains，这个字段就设为 None，表示不适用。
    forbidden_answer_hit = len(forbidden_hits) > 0 # 只要命中一个 forbidden_answer_contains 就算 hit，这样可以更严格地控制答案中不应该出现的内容。

    return {
        "expected_answer_contains": expected_answer_contains,
        "forbidden_answer_contains": forbidden_answer_contains,
        "expected_hits": expected_hits,
        "forbidden_hits": forbidden_hits,
        "expected_answer_hit": expected_answer_hit,
        "forbidden_answer_hit": forbidden_answer_hit,
    }

def evalvate_answer_sources(sample: dict, answer_sources: list[str]) -> dict:
    """评估答案中的 sources 是否包含期望的关键引用。"""
    expected_sources_contains = sample.get("expected_sources_contains", [])

    normalized_sources = [strip_chunk_id(source) for source in answer_sources]

    source_hits = [
        item for item in expected_sources_contains
        if any(item in source for source in normalized_sources)
    ]

    source_citation_hit = (
        len(source_hits) == len(expected_sources_contains)
        if expected_sources_contains else None
    )

    return {
        "expected_sources_contains": expected_sources_contains,
        "source_hits": source_hits,
        "source_citation_hit": source_citation_hit,
    }



def evaluate_agent_sample(sample: dict, answer_result) -> dict:
    """评估真实 ask_course_agent 结果，包括 sources 和 answer 文本。"""
    evaluated = evaluate_sources(
        sample,
        answer_result.sources,
    )
    answer_eval = evaluate_answer_content(sample, answer_result.answer)

    source_eval = evalvate_answer_sources(sample, answer_result.sources)
    evaluated.update(
        {
            "id": sample["id"],
            "task_type": sample["task_type"],
            "question": sample["question"],
            "answer_preview": answer_result.answer[:200],
            "suggestion_count": len(answer_result.suggestions),
            "expected_answer_contains": answer_eval["expected_answer_contains"],
            "forbidden_answer_contains": answer_eval["forbidden_answer_contains"],
            "expected_hits": answer_eval["expected_hits"],
            "forbidden_hits": answer_eval["forbidden_hits"],
            "expected_answer_hit": answer_eval["expected_answer_hit"],
            "forbidden_answer_hit": answer_eval["forbidden_answer_hit"],
            "expected_sources_contains": source_eval["expected_sources_contains"],
            "source_hits": source_eval["source_hits"],
            "source_citation_hit": source_eval["source_citation_hit"],
        }
    )
    return evaluated

def append_eval_trace(payload: dict) -> None:
    """把评估过程中的检索 trace 写入到文件，方便后续分析和调试。"""
    EVAL_TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)

    with EVAL_TRACE_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")

def build_eval_trace_payload(
    mode: str,
    sample: dict,
    retrieval_eval: dict,
    agent_eval: dict | None = None,
    answer_result=None,
) -> dict:
    """构建一个评估 trace 的 payload，包含样本基本信息、检索评估结果、以及如果有的话，答案评估结果和模型输出的相关信息。这个 payload 会被写入到 eval_traces.jsonl 文件中，供后续分析和调试使用。"""
    payload = {
        "sample_id": sample["id"],
        "mode": mode,
        "task_type": sample["task_type"],
        "question": sample["question"],
        "retrieval_eval": retrieval_eval,
    }

    if answer_result is not None:
        payload["answer_preview"] = answer_result.answer[:200]
        payload["answer_sources"] = answer_result.sources
        payload["agent_debug"] = getattr(answer_result, "debug", {})

    if agent_eval is not None:
        payload["agent_eval"] = agent_eval

    return payload

def should_run_agent_sample(sample: dict) -> bool:
    """控制哪些样本进入真实 Agent Eval，避免一次性把所有题都打到模型。"""
    allowed_ids = {
        "study-plan-agent-001",
        "study-plan-scope-001",
        "study-plan-rag-001",
        "qa-tool-use-001",
        "qa-agentic-rag-001",
        "summary-tool-use-001",
        "summary-agentic-rag-001",
    }
    return sample["id"] in allowed_ids


def print_mode_summary(mode: str, results: list[dict]) -> None:
    total = len(results)
    primary_hits = sum(1 for item in results if item["primary_hit"])
    group_hits = sum(1 for item in results if item["group_hit"])
    summary_items = [item for item in results if item["task_type"] == "summary"]
    summary_pollutions = sum(1 for item in summary_items if item["summary_polluted"])
    coverage_items = [
        item for item in results
        if item["group_coverage"] is not None
    ]

    print(f"\n=== 检索模式: {mode} ===")
    if coverage_items:
        average_group_coverage = sum(
            item["group_coverage"] for item in coverage_items
        ) / len(coverage_items)
        print(f"平均课程簇覆盖率: {average_group_coverage:.2f}")
    print(f"样本总数: {total}")
    print(f"首条来源命中: {primary_hits}/{total}")
    print(f"课程簇命中: {group_hits}/{total}")

    if summary_items:
        print(
            f"总结污染数: {summary_pollutions}/{len(summary_items)} "
            f"(越低越好)"
        )


def print_detailed_results(mode: str, results: list[dict]) -> None:
    print(f"\n--- 详细结果: {mode} ---")
    for item in results:
        print(f"\n[{item['id']}] {item['question']}")
        print(f"  首条来源: {item['first_source']}")
        print(f"  首条来源命中: {item['primary_hit']}")
        print(f"  首条课程簇: {item['first_group']}")
        print(f"  课程簇命中: {item['group_hit']}")
        print(f"  是否总结污染: {item['summary_polluted']}")
        if item["group_coverage"] is not None:
            print(f"  命中的课程簇: {item['covered_groups']}")
            print(f"  课程簇覆盖率: {item['group_coverage']:.2f}")
        print("  来源列表:")
        for source in item["sources"]:
            print(f"    - {source}")


def print_agent_summary(mode: str, results: list[dict]) -> None:
    total = len(results)
    if total == 0:
        return

    primary_hits = sum(1 for item in results if item["primary_hit"])
    group_hits = sum(1 for item in results if item["group_hit"])
    coverage_items = [
        item for item in results
        if item["group_coverage"] is not None
    ]
    average_group_coverage = None
    if coverage_items:
        average_group_coverage = sum(
            item["group_coverage"] for item in coverage_items
        ) / len(coverage_items)

    # 评估答案内容的命中情况，主要关注 expected_answer_hit 和 forbidden_answer_hit 这两个字段。
    # expected_answer_hit 表示答案是否包含了所有期望出现的内容，而 forbidden_answer_hit 表示答案是否包含了不该出现的内容。
    # 理想情况下，我们希望 expected_answer_hit 是 True，forbidden_answer_hit 是 False。
    answer_hit_items = [item for item in results if item["expected_answer_hit"] is not None]
    answer_hits = sum(1 for item in answer_hit_items if item["expected_answer_hit"])
    forbidden_hits = sum(1 for item in results if item["forbidden_answer_hit"])

    # 来源引用指标
    source_hit_items = [
        item for item in results
        if item["source_citation_hit"] is not None
    ]
    source_hits = sum(1 for item in source_hit_items if item["source_citation_hit"])

    print(f"\n=== Agent 评估模式: {mode}（selected samples）===")
    print(f"样本总数: {total}")
    print(f"首条来源命中: {primary_hits}/{total}")
    print(f"课程簇命中: {group_hits}/{total}")
    print("主指标: Expected Answer Hit / Forbidden Answer Hit")
    if average_group_coverage is not None:
        print(f"平均课程簇覆盖率: {average_group_coverage:.2f}")

    if answer_hit_items:
        print(f"期望答案命中: {answer_hits}/{len(answer_hit_items)} (越高越好)")
    print(f"禁用答案命中: {forbidden_hits}/{total} (越低越好)")

    if source_hit_items:
        print(f"来源引用命中: {source_hits}/{len(source_hit_items)}(越高越好)")


def print_agent_detailed_results(mode: str, results: list[dict]) -> None:
    if not results:
        return

    print(f"\n--- Agent 详细结果: {mode} ---")
    for item in results:
        print(f"\n[{item['id']}] {item['question']}")
        print(f"  首条来源: {item['first_source']}")
        print(f"  首条来源命中: {item['primary_hit']}")
        print(f"  首条课程簇: {item['first_group']}")
        print(f"  课程簇命中: {item['group_hit']}")
        if item["group_coverage"] is not None:
            print(f"  命中的课程簇: {item['covered_groups']}")
            print(f"  课程簇覆盖率: {item['group_coverage']:.2f}")
        print(f"  建议条数: {item['suggestion_count']}")
        if item["expected_answer_hit"] is not None: # 只有当 sample 中有 expected_answer_contains 时，这个字段才有意义，我们才打印相关信息。
            print(f"  期望答案命中: {item['expected_answer_hit']}")
            print(f"  期望命中项: {item['expected_hits']}")
        print(f"  禁用答案命中: {item['forbidden_answer_hit']}")
        if item["forbidden_hits"]: # 只有当实际命中了 forbidden_answer_contains 时，这个列表才有内容，我们才打印相关信息。
            print(f"  禁用命中项: {item['forbidden_hits']}")
        if item["source_citation_hit"] is not None:
            print(f"  来源引用命中: {item['source_citation_hit']}")
            print(f"  来源命中项: {item['source_hits']}")
        print(f"  回答预览: {item['answer_preview']}")
        print("  来源列表:")
        for source in item["sources"]:
            print(f"    - {source}")


def main():
    settings = get_settings()
    run_agent_eval = should_run_agent_eval()
    modes = get_eval_modes()
    active_tags = get_eval_tags()
    questions = load_questions()
    documents = load_documents(
        settings.course_source_root,
        include_dirs=settings.course_include_dirs,
    )
    chunks = load_document_chunks(
        settings.course_source_root,
        include_dirs=settings.course_include_dirs,
    )

    print(f"已加载文档数: {len(documents)}")
    print(f"已加载切块数: {len(chunks)}")
    print(f"已加载评估题数: {len(questions)}")
    print(f"评估模式 EVAL_MODES={','.join(modes)}")
    print(f"评估标签 EVAL_TAGS={','.join(sorted(active_tags)) if active_tags else 'all'}")
    print(f"是否运行 Agent Eval={run_agent_eval}")

    # 控制是否写评估 trace，以及如果写的话，写到哪里。
    write_eval_traces = should_write_eval_traces()
    print(f"是否写入评估 Trace={write_eval_traces}")
    # 如果启用了写评估 trace，并且之前的 trace 文件存在，就先删除它，确保本次评估的 trace 是干净的，不会和之前的结果混在一起，方便后续分析和调试。
    if write_eval_traces and EVAL_TRACE_PATH.exists():
        EVAL_TRACE_PATH.unlink()

    modes = get_eval_modes()
    all_results: dict[str, list[dict]] = {}
    all_agent_results: dict[str, list[dict]] = {}

    vector_store = None
    reranker = None

    if "vector" in modes or "hybrid" in modes:
        vector_store = build_vector_store_with_cache(settings, chunks)

    if "hybrid" in modes:
        reranker = build_active_reranker(settings)

    for mode in modes:
        mode_settings = replace(
            settings,
            retrieval=replace(settings.retrieval, retrieval_mode=mode),
        )
        mode_results: list[dict] = []
        mode_agent_results: list[dict] = []

        for sample in questions:
            if not is_tag_enabled_for_sample(sample, active_tags):
                continue

            if not is_mode_enabled_for_sample(sample, mode):# 如果当前样本没有启用这个 mode，就跳过，不参与评估。这样我们就可以在 questions.json 中灵活地控制每个样本在哪些 mode 下参与评估，避免一些不合理的样本对某些 mode 造成干扰，同时也能更专注地分析每个 mode 的表现。
                continue

            sample_memory = sample.get("memory")

            print(f"\n正在运行评估模式: {mode}")
            retrieved = retrieve_for_mode(
                question=sample["question"],
                mode=mode,
                settings=mode_settings,
                documents=documents,
                chunks=chunks,
                vector_store=vector_store,
                memory=sample_memory,
                reranker=reranker if mode == "hybrid" else None,
            )
            evaluated = evaluate_sample(sample, retrieved)
            mode_results.append(evaluated)

            # 评估完检索结果后，先把这个评估结果写入 trace 文件，方便我们在后续分析时，能看到每个样本在每个 mode 下的检索表现，以及它们的来源情况。
            # 这些信息对于我们理解模型的行为、分析失败案例、以及指导后续的优化方向，都非常有价值。
            if write_eval_traces and not (run_agent_eval and should_run_agent_sample(sample)):
                trace_payload = build_eval_trace_payload(
                    mode=mode,
                    sample=sample,
                    retrieval_eval=evaluated,
                )
                append_eval_trace(trace_payload)

            # 控制哪些样本进入真实 Agent Eval，避免一次性把所有题都打到模型，造成不必要的成本和等待时间。
            # 我们可以通过 sample 的 id 来控制，只有那些 id 在 should_run_agent_sample 函数里明确列出的样本，才会进入真实的 ask_course_agent 调用和评估。
            # 这样我们就可以先选一些典型的样本进行初步评估和分析，等我们对模型的表现有了更深入的理解之后，再逐步放开更多的样本进行全面评估。
            if run_agent_eval and should_run_agent_sample(sample):
                answer_result = ask_course_agent(
                    question=sample["question"],
                    documents=documents,
                    settings=mode_settings,
                    memory=sample_memory,
                    chunks=chunks,
                    vector_store=vector_store,
                    reranker=reranker if mode == "hybrid" else None,
                )
                evaluated_agent = evaluate_agent_sample(sample, answer_result)
                mode_agent_results.append(evaluated_agent)

                if write_eval_traces:
                    trace_payload = build_eval_trace_payload(
                        mode=mode,
                        sample=sample,
                        retrieval_eval=evaluated,
                        agent_eval=evaluated_agent,
                        answer_result=answer_result,
                    )
                    append_eval_trace(trace_payload)

        all_results[mode] = mode_results
        all_agent_results[mode] = mode_agent_results
        print_mode_summary(mode, mode_results)
        print_agent_summary(mode, mode_agent_results)

    for mode in modes:
        print_detailed_results(mode, all_results[mode])
        print_agent_detailed_results(mode, all_agent_results[mode])


if __name__ == "__main__":
    main()
