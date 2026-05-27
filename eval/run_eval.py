import json
import os
import sys
from dataclasses import replace
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = PROJECT_ROOT / "app"

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

def strip_chunk_id(source: str) -> str:
    return source.split("#", 1)[0]


def get_course_group(source: str) -> str | None:
    parts = Path(source).parts
    for part in parts:
        if part.startswith(("1-", "2-")):
            return part
    return None

def build_active_reranker(settings):
    if settings.reranker_provider == "none":
        return None

    return build_reranker(
        settings.reranker_provider,
        settings.reranker_model_name,
        settings.reranker_cache_dir,
    )

def should_run_agent_eval() -> bool:
    """读取 RUN_AGENT_EVAL 开关，控制是否执行真实模型调用。"""
    raw_value = os.getenv("RUN_AGENT_EVAL", "false").strip().lower()
    return raw_value not in {"0", "false", "no", "off"}

def get_eval_modes() -> list[str]:
    """读取 EVAL_MODES 开关， 控制本次评估运行哪些检索模式。"""
    raw_value = os.getenv("EVAL_MODES", "chunk,vector,hybrid").strip().lower()
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
    raw_value = os.getenv("EVAL_TAGS", "").strip().lower()
    if not raw_value:
        return None

    tags = {item.strip() for item in raw_value.split(",") if item.strip()}
    return tags or None


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
            top_k=settings.retrieval_top_k,
        )
    elif mode == "chunk":
        retrieved = retrieve_chunks(
            query=question,
            chunks=chunks,
            top_k=settings.retrieval_top_k,
        )
    elif mode == "vector":
        if vector_store is None:
            raise ValueError("vector mode requires vector_store")
        retrieved = vector_store.search(
            query=question,
            top_k=settings.retrieval_top_k,
        )
    elif mode == "hybrid":
        if vector_store is None:
            raise ValueError("hybrid mode requires vector_store")

        candidate_top_k = max(settings.retrieval_top_k * 3, 10)

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
            top_k=settings.retrieval_top_k,
        )
        if reranker is not None:
            retrieved = reranker.rerank(
                question,
                retrieved,
                top_k=settings.retrieval_top_k,
            )
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    task_type = detect_task_type(question)
    if task_type == "summary":
        retrieved = narrow_summary_results(retrieved)

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



def evaluate_agent_sample(sample: dict, answer_result) -> dict:
    """评估真实 ask_course_agent 结果，包括 sources 和 answer 文本。"""
    evaluated = evaluate_sources(
        sample,
        answer_result.sources,
    )
    answer_eval = evaluate_answer_content(sample, answer_result.answer)
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
        }
    )
    return evaluated


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

    print(f"\n=== Mode: {mode} ===")
    if coverage_items:
        average_group_coverage = sum(
            item["group_coverage"] for item in coverage_items
        ) / len(coverage_items)
        print(f"Average Group Coverage: {average_group_coverage:.2f}")
    print(f"Total: {total}")
    print(f"Primary Hit: {primary_hits}/{total}")
    print(f"Group Hit: {group_hits}/{total}")

    if summary_items:
        print(
            f"Summary Pollution: {summary_pollutions}/{len(summary_items)} "
            f"(越低越好)"
        )


def print_detailed_results(mode: str, results: list[dict]) -> None:
    print(f"\n--- Detailed Results: {mode} ---")
    for item in results:
        print(f"\n[{item['id']}] {item['question']}")
        print(f"  First Source: {item['first_source']}")
        print(f"  Primary Hit: {item['primary_hit']}")
        print(f"  First Group: {item['first_group']}")
        print(f"  Group Hit: {item['group_hit']}")
        print(f"  Summary Polluted: {item['summary_polluted']}")
        if item["group_coverage"] is not None:
            print(f"  Covered Groups: {item['covered_groups']}")
            print(f"  Group Coverage: {item['group_coverage']:.2f}")
        print("  Sources:")
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

    print(f"\n=== Agent Mode: {mode} (study_plan only) ===")
    print(f"Total: {total}")
    print(f"Primary Hit: {primary_hits}/{total}")
    print(f"Group Hit: {group_hits}/{total}")
    print("Main Metric: Expected Answer Hit / Forbidden Answer Hit")
    if average_group_coverage is not None:
        print(f"Average Group Coverage: {average_group_coverage:.2f}")

    if answer_hit_items:
        print(f"Expected Answer Hit: {answer_hits}/{len(answer_hit_items)} (越高越好)")
    print(f"Forbidden Answer Hit: {forbidden_hits}/{total} (越低越好)")


def print_agent_detailed_results(mode: str, results: list[dict]) -> None:
    if not results:
        return

    print(f"\n--- Agent Detailed Results: {mode} ---")
    for item in results:
        print(f"\n[{item['id']}] {item['question']}")
        print(f"  First Source: {item['first_source']}")
        print(f"  Primary Hit: {item['primary_hit']}")
        print(f"  First Group: {item['first_group']}")
        print(f"  Group Hit: {item['group_hit']}")
        if item["group_coverage"] is not None:
            print(f"  Covered Groups: {item['covered_groups']}")
            print(f"  Group Coverage: {item['group_coverage']:.2f}")
        print(f"  Suggestion Count: {item['suggestion_count']}")
        if item["expected_answer_hit"] is not None: # 只有当 sample 中有 expected_answer_contains 时，这个字段才有意义，我们才打印相关信息。
            print(f"  Expected Answer Hit: {item['expected_answer_hit']}")
            print(f"  Expected Hits: {item['expected_hits']}")
        print(f"  Forbidden Answer Hit: {item['forbidden_answer_hit']}")
        if item["forbidden_hits"]: # 只有当实际命中了 forbidden_answer_contains 时，这个列表才有内容，我们才打印相关信息。
            print(f"  Forbidden Hits: {item['forbidden_hits']}")
        print(f"  Answer Preview: {item['answer_preview']}")
        print("  Sources:")
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

    print(f"Loaded {len(documents)} documents")
    print(f"Loaded {len(chunks)} chunks")
    print(f"Loaded {len(questions)} eval questions")
    print(f"EVAL_MODES={','.join(modes)}")
    print(f"EVAL_TAGS={','.join(sorted(active_tags)) if active_tags else 'all'}")
    print(f"RUN_AGENT_EVAL={run_agent_eval}")

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
        mode_settings = replace(settings, retrieval_mode=mode)
        mode_results: list[dict] = []
        mode_agent_results: list[dict] = []

        for sample in questions:
            if not is_tag_enabled_for_sample(sample, active_tags):
                continue

            if not is_mode_enabled_for_sample(sample, mode):# 如果当前样本没有启用这个 mode，就跳过，不参与评估。这样我们就可以在 questions.json 中灵活地控制每个样本在哪些 mode 下参与评估，避免一些不合理的样本对某些 mode 造成干扰，同时也能更专注地分析每个 mode 的表现。
                continue

            sample_memory = sample.get("memory")

            print(f"\nRunning mode: {mode}")
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

            if run_agent_eval and sample["task_type"] == "study_plan":
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

        all_results[mode] = mode_results
        all_agent_results[mode] = mode_agent_results
        print_mode_summary(mode, mode_results)
        print_agent_summary(mode, mode_agent_results)

    for mode in modes:
        print_detailed_results(mode, all_results[mode])
        print_agent_detailed_results(mode, all_agent_results[mode])


if __name__ == "__main__":
    main()
