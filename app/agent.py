# agent.py：调模型
# 串主流程：读取问题、调检索、组装上下文、调模型、返回结构化结果
import json
from vector_store import VectorStore
from reranker import Reranker

# pyrefly: ignore [missing-import]
from openai import OpenAI
# pyrefly: ignore [missing-import]
from pydantic import ValidationError
from pathlib import Path

from config import Settings, get_settings, validate_settings
from prompts import (
    SYSTEM_PROMPT,
    build_user_prompt,
    build_summary_prompt,
    build_study_plan_prompt,
)
from retriever import retrieve_documents, retrieve_chunks
from schemas import AgentAnswer, Document, DocumentChunk, RetrievedChunk

STUDY_PLAN_ORDER_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "study_plan_order.json"
)
AGENT_RUNTIME_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "agent_runtime_config.json"
)

def build_client(settings: Settings) -> OpenAI:
    client_kwargs = {"api_key": settings.api_key}

    if settings.base_url:
        client_kwargs["base_url"] = settings.base_url

    return OpenAI(**client_kwargs)

def load_study_plan_order_config() -> dict:
    """读取 study_plan 排序配置。"""
    with STUDY_PLAN_ORDER_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_agent_runtime_config() -> dict:
    with AGENT_RUNTIME_CONFIG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)

# 这个函数的作用是从给定的 source 路径中提取课程组信息。我们假设课程组的信息包含在路径的某个部分，并且以 "1-" 或 "2-" 开头。通过这个函数，我们可以在后续的检索结果排序中优先展示与用户问题相关的课程组内容，从而提升用户体验和检索结果的相关性。
def get_course_group(source: str) -> str | None:
    parts = Path(source).parts
    for part in parts:
        if part.startswith(("1-", "2-")):
            return part
    return None

# 这个函数的作用是对检索到的结果进行过滤和排序，优先保留与首条结果来自同一课程组的内容。这样做的好处是可以提升检索结果的相关性和一致性，尤其是在用户提问中包含了特定课程组信息的情况下。通过这种方式，我们可以更好地满足用户的查询意图，提供更精准和有针对性的回答。
def narrow_summary_results(
    retrieved_chunks: list[RetrievedChunk],
    strategy: str = "same-source",
) -> list[RetrievedChunk]:
    if not retrieved_chunks:
        return []

    if strategy == "same-source":
        target_source = retrieved_chunks[0].source
        same_source = [
            item for item in retrieved_chunks
            if item.source == target_source
        ]
        return same_source or retrieved_chunks

    return retrieved_chunks

# 混合检索 用 retrieve_chunks(...) 取一份结果，用 vector_store.search(...) 再取一份结果，然后把两份结果合并去重，最终返回 top_k 条结果。这样做的好处是：
# 1. retrieve_chunks(...) 的结果通常更精准，因为它直接基于文本内容进行匹配，能够捕捉到一些细粒度的相关信息；而 vector_store.search(...) 的结果可能更全面，因为它基于向量表示进行匹配。通过混合检索，我们可以兼顾精准性和全面性，提升整体的检索效果。
# 2. retrieve_chunks(...) 的结果可以作为 vector_store.search(...) 的补充，当 retrieve_chunks(...) 没有检索到足够的相关内容时，vector_store.search(...) 可以提供更多的候选项，增加找到相关信息的机会。反过来，当 retrieve_chunks(...) 已经检索到足够的相关内容时，我们也可以通过混合检索来引入一些 vector_store.search(...) 的结果，增加多样性和覆盖面。
def merge_retrieval_results(
    primary: list[RetrievedChunk],
    secondary: list[RetrievedChunk],
    top_k: int,
    max_per_source: int | None = None,
) -> list[RetrievedChunk]:
    runtime_config = load_agent_runtime_config()
    merge_config = runtime_config.get("retrieval_merge", {})
    if max_per_source is None:
        max_per_source = merge_config.get("max_per_source", 2)
    prioritize_same_group_secondary = merge_config.get(
        "prioritize_same_group_secondary",
        True,
    )

    merged: list[RetrievedChunk] = []
    seen: set[tuple[str, str | None]] = set() # 用于去重，记录已经添加过的 (source, chunk_id) 组合
    source_counts: dict[str, int] = {} # 记录每个 source 已经添加了多少条结果 (同源限流)

    primary_group = get_course_group(primary[0].source) if primary else None

    same_group_secondary: list[RetrievedChunk] = []
    other_group_secondary: list[RetrievedChunk] = []
    # 如果首条来自：2-3-ai-agents-for-beginners. 那后面的补充结果会优先保留同属 2-3 的模块，再把 2-2 的结果往后放
    for item in secondary:# 先把 secondary 里的结果根据是否与 primary 的课程组相同分成两类，这样我们就可以在后续的排序中优先保留同组的结果，进一步提升相关性
        if primary_group and get_course_group(item.source) == primary_group: 
            same_group_secondary.append(item)
        else:
            other_group_secondary.append(item)

    if prioritize_same_group_secondary:
        candidates = primary + same_group_secondary + other_group_secondary
    else:
        candidates = primary + secondary

    for item in candidates: # 按照 primary 结果优先、同组 secondary 结果次之、其他 secondary 结果最后的顺序来遍历候选项，依次添加到 merged 结果中，同时进行去重和同源限流，直到达到 top_k 条结果为止。
        key = (item.source, item.chunk_id)
        if key in seen:
            continue

        count = source_counts.get(item.source, 0)
        if count >= max_per_source: # 同一个 source 最多保留 2 条, 仍然优先保留 primary 里的结果,secondary 只作为补充
            continue

        seen.add(key)
        merged.append(item)
        source_counts[item.source] = count + 1

        if len(merged) >= top_k:
            break

    return merged

def merge_multi_query_results(
        result_groups: list[list[RetrievedChunk]],
        top_k: int,
) -> list[RetrievedChunk]:
    """合并多条 query 检索结果；相同chunk保留最高分。"""
    best_by_key: dict[tuple[str, str | None], RetrievedChunk] = {}

    for group in result_groups:
        for item in group:
            key = (item.source, item.chunk_id)
            existing = best_by_key.get(key)
            if existing is None or item.score > existing.score:
                best_by_key[key] = item
    merged = sorted(
        best_by_key.values(),
        key=lambda item: item.score,
        reverse=True
    )
    return merged[:top_k]

def should_retry_retrieval(
        retrieved_chunks: list[RetrievedChunk],
        task_type: str,
) -> bool:
    """
    判断结果是否弱
    弱的话，做第二轮更强检索
        qa 少于 2 条就算弱
        summary 少于 2 条就算弱
        study_plan 少于 3 条就算弱
    """
    if not retrieved_chunks:
        return True
    
    if task_type == "summary":
        return len(retrieved_chunks) < 2
    
    if task_type == "study_plan":
        return len(retrieved_chunks) < 3
    
    return len(retrieved_chunks) < 2

# 根据任务类型选择要放入 prompt 的检索结果数量和策略。对于总结类问题，我们可能只需要最相关的 3-4 条资料来生成高质量的总结；对于学习计划类问题，我们可能需要更多的资料来全面评估和安排学习路线；而对于一般的问答类问题，我们可能只需要最相关的 1-3 条资料来直接回答用户的问题。
def select_prompt_chunks(task_type: str, retrieved_chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    limits = {
        "qa": 3,
        "summary": 4,
        "study_plan": 5,
    }
    limit = limits.get(task_type, 3)
    return retrieved_chunks[:limit]

# 这个函数的作用是从一组 RetrievedChunk 中提取出它们的来源引用，并返回一个列表。我们通过 build_source_reference 函数把每个 chunk 转换成一个唯一的来源引用字符串，然后去重后返回。这个列表可以用来填充模型输出中的 sources 字段，确保它们与实际使用过的检索结果对应起来。
def build_source_reference_list(chunks: list[RetrievedChunk]) -> list[str]:
    seen: set[str] = set()
    references: list[str] = []
    for chunk in chunks:
        source = format_source_reference(chunk)
        if source in seen:
            continue
        seen.add(source)
        references.append(source)
    return references

# 这个函数的作用是对模型输出中的 sources 字段进行规范化处理，确保它们只包含 allowed_sources 中的真实来源，并且去重后返回。如果模型输出中的 sources 字段包含了未使用过的来源或者重复的来源，我们就过滤掉它们，最终返回一个干净、准确的来源列表。如果过滤掉之后没有任何合法来源了，我们就返回一个 fallback_sources 作为兜底，确保最终返回给用户的答案中至少有一些合理的来源信息。
def normalize_answer_sources(answer_sources: list[str], allowed_sources: list[str], fallback_sources: list[str]) -> list[str]:
    allowed_set = set(allowed_sources)
    normalized: list[str] = []
    seen: set[str] = set()
    for source in answer_sources:
        if source not in allowed_set or source in seen:
            continue
        seen.add(source)
        normalized.append(source)
    return normalized or fallback_sources

def normalize_suggestions(
    suggestions: list[str],
    question: str,
    max_items: int = 3,
) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    question_text = question.strip()

    for item in suggestions:
        text = item.strip()
        if not text:
            continue
        if text == question_text:
            continue
        if text in seen:
            continue
        seen.add(text)
        normalized.append(text)

        if len(normalized) >= max_items:
            break

    return normalized

def normalize_answer_text(answer_text: str) -> str:
    text = answer_text.strip()

    if not text:
        return "根据当前资料，暂时无法生成稳定答案。"

    return text

def ask_course_agent(
    question: str, 
    documents: list[Document], 
    settings: Settings | None = None, 
    memory: dict | None = None, 
    chunks: list[DocumentChunk] | None = None,
    vector_store: VectorStore | None = None,
    reranker: Reranker | None = None,
    ) -> AgentAnswer:
    active_settings = settings or get_settings()
    validate_settings(active_settings)
    task_type = detect_task_type(question)
    retrieval_queries = build_retrieval_queries(
        question,
        task_type,
        memory=memory,
    )

    debug_info = {
        "task_type": task_type,
        "initial_queries": retrieval_queries,
        "retry_triggered": False,
        "retry_queries": [],
        "llm_retry_query": None,
        "initial_result_count": 0,
        "final_result_count": 0,
    }

    retrieved_chunks = run_retrieval_round(
        retrieval_queries=retrieval_queries,
        active_settings=active_settings,
        documents=documents,
        chunks=chunks,
        vector_store=vector_store,
        reranker=reranker,
    )
    debug_info["initial_result_count"] = len(retrieved_chunks)

    client = build_client(active_settings)

    if should_retry_retrieval(retrieved_chunks, task_type):# 弱的话，做第二轮更强检索
        debug_info["retry_triggered"] = True

        retry_queries = build_retry_retrieval_queries(
            question,
            task_type,
            memory=memory,
            settings=active_settings,
            client=client,
        )
        debug_info["retry_queries"] = retry_queries

        # 记录模型到底加了什么（多生成1条LLM补强的query）
        base_retry_queries = dedupe_queries(
            [
                f"{question.strip()} notebook lesson summary"
                if task_type == "summary"
                else f"{question.strip()} 学习顺序 roadmap lesson"
                if task_type == "study_plan"
                else f"{question.strip()} agent course concept"
            ]
            + build_anchor_queries(question)
            + build_memory_queries(task_type, memory=memory)
        )
        extra_retry_queries = [
            item for item in retry_queries
            if item not in base_retry_queries
        ]
        if extra_retry_queries:
            debug_info["llm_retry_query"] = extra_retry_queries[-1]

        if retry_queries:
            retry_chunks = run_retrieval_round(
                retrieval_queries=retry_queries,
                active_settings=active_settings,
                documents=documents,
                chunks=chunks,
                vector_store=vector_store,
                reranker=reranker,
            )
            retrieved_chunks = merge_multi_query_results(
                [retrieved_chunks, retry_chunks],
                top_k=active_settings.retrieval.retrieval_top_k,
            )

    if not retrieved_chunks:
        runtime_config = load_agent_runtime_config()
        fallback_config = runtime_config.get("fallback", {})
        return AgentAnswer(
            answer=fallback_config.get(
                "no_results_answer",
                "当前没有检索到相关课程资料，暂时无法回答这个问题",
            ),
            suggestions=fallback_config.get(
                "no_results_suggestions",
                ["换一个更具体的问题试试", "优先使用课程名称、章节名或关键词提问"],
            ),
            sources=[]
        )

    if task_type == "summary":
        retrieved_chunks = narrow_summary_results(
            retrieved_chunks,
            strategy=active_settings.retrieval.summary_strategy,
        )
    if task_type == "study_plan":
        retrieved_chunks = post_rank_study_plan_results(question, retrieved_chunks)
    
    debug_info["final_result_count"] = len(retrieved_chunks)

    prompt_chunks = select_prompt_chunks(task_type, retrieved_chunks)
    allowed_sources = build_source_reference_list(prompt_chunks)
    fallback_sources = allowed_sources[:3] # 最多保留前三条作为兜底来源，确保即使模型输出的 sources 字段完全不合法，我们也能返回一些合理的来源信息

    if task_type == "summary":
        user_prompt = build_summary_prompt(question, prompt_chunks, memory=memory)
    elif task_type == "study_plan":
        user_prompt = build_study_plan_prompt(question, prompt_chunks, memory=memory)
    else:
        user_prompt = build_user_prompt(question, prompt_chunks, memory=memory)

    if task_type == "summary" and memory is not None and retrieved_chunks:
        # 第一条通常就是最相关、最可能是这次总结目标的章节 
        # 为什么这里不直接 save_user_memory(...)
            #因为现在更好的职责分层是：
            #agent.py 负责修改内存中的 memory
            #main.py 负责在合适的时候保存到文件
        update_completed_topic(memory, retrieved_chunks[0].title)

    response = client.chat.completions.create(
        model=active_settings.model_name,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    raw_content = response.choices[0].message.content or "{}"

    try:
        payload = json.loads(raw_content)
        answer = AgentAnswer.model_validate(payload)
    except (json.JSONDecodeError, ValidationError):
        answer = AgentAnswer(
            answer=raw_content,
            suggestions=[],
            sources=fallback_sources,
            debug=debug_info,
        )
    
    answer.answer = normalize_answer_text(answer.answer)
    answer.suggestions = normalize_suggestions(
        answer.suggestions,
        question=question,
    )
    answer.sources = normalize_answer_sources(
        answer.sources,
        allowed_sources=allowed_sources,
        fallback_sources=fallback_sources,
    )
    answer.debug = debug_info

    return answer

def run_retrieval_round(
    retrieval_queries: list[str],
    active_settings,
    documents: list[Document],
    chunks: list[DocumentChunk] | None = None,
    vector_store: VectorStore | None = None,
    reranker: Reranker | None = None,
) -> list[RetrievedChunk]:
    if active_settings.retrieval.retrieval_mode == "chunk":
        if chunks is not None:
            result_groups = [
                retrieve_chunks(
                    query=query,
                    chunks=chunks,
                    top_k=active_settings.retrieval.retrieval_top_k,
                )
                for query in retrieval_queries
            ]
        else:
            result_groups = [
                retrieve_documents(
                    query=query,
                    documents=documents,
                    top_k=active_settings.retrieval.retrieval_top_k,
                )
                for query in retrieval_queries
            ]

        return merge_multi_query_results(
            result_groups,
            top_k=active_settings.retrieval.retrieval_top_k,
        )

    if active_settings.retrieval.retrieval_mode == "document":
        result_groups = [
            retrieve_documents(
                query=query,
                documents=documents,
                top_k=active_settings.retrieval.retrieval_top_k,
            )
            for query in retrieval_queries
        ]
        return merge_multi_query_results(
            result_groups,
            top_k=active_settings.retrieval.retrieval_top_k,
        )

    if active_settings.retrieval.retrieval_mode == "hybrid":# 混合检索
        """
            每个 query 都做一轮 lexical + vector + merge
            然后多 query 再 merge
            最后 rerank 一次
        """
        if chunks is None:
            raise ValueError("chunk mode requires chunks")
        if vector_store is None:
            raise ValueError("hybrid mode requires vector_store")

        candidate_top_k = max(
            active_settings.retrieval.retrieval_top_k
            * active_settings.retrieval.hybrid_candidate_multiplier,
            active_settings.retrieval.hybrid_candidate_minimum,
        )

        hybrid_groups: list[list[RetrievedChunk]] = []

        for query in retrieval_queries:
            # 这里先让 lexical_results 放前面，是因为：标题/关键词精确命中对课程问答很重要    vector 先作为补充召回
            lexical_results = retrieve_chunks(
                query=query,
                chunks=chunks,
                top_k=candidate_top_k,
            )
            vector_results = vector_store.search(
                query=query,
                top_k=candidate_top_k,
            )
            merged_results = merge_retrieval_results(
                lexical_results,
                vector_results,
                top_k=active_settings.retrieval.retrieval_top_k,
            )
            hybrid_groups.append(merged_results)

        retrieved_chunks = merge_multi_query_results(
            hybrid_groups,
            top_k=active_settings.retrieval.retrieval_top_k,
        )# lexical 定主轴，vector 做补充

        if reranker is not None:# 如果提供了 reranker，就对混合检索的结果进行重新排序，进一步提升相关性。这里我们把混合检索的结果作为 reranker 的输入，让它根据查询和每条结果的内容来打分排序，从而把最相关的结果排在前面，提升最终返回给用户的答案的质量和准确性。
            retrieved_chunks = reranker.rerank(
                retrieval_queries[0],
                retrieved_chunks,
                top_k=active_settings.retrieval.retrieval_top_k,
            )

        return retrieved_chunks

    if active_settings.retrieval.retrieval_mode == "vector":
        if vector_store is None:
            raise ValueError("vector mode requires vector_store")

        result_groups = [
            vector_store.search(
                query=query,
                top_k=active_settings.retrieval.retrieval_top_k,
            )
            for query in retrieval_queries
        ]
        return merge_multi_query_results(
            result_groups,
            top_k=active_settings.retrieval.retrieval_top_k,
        )

    raise ValueError(f"Unsupported retrieval mode: {active_settings.retrieval.retrieval_mode}")


def detect_task_type(question: str) -> str:
    lowered = question.lower()
    runtime_config = load_agent_runtime_config()
    task_routing = runtime_config.get("task_routing", {})
    summary_keywords = task_routing.get(
        "summary_keywords",
        [
            "总结",
            "概述",
            "概要",
            "讲什么",
            "这一节",
            "这节课",
            "notebook",
            "lesson",
        ],
    )
    study_plan_keywords = task_routing.get(
        "study_plan_keywords",
        [
            "学习顺序",
            "学习路线",
            "学习计划",
            "怎么学",
            "从哪里开始",
            "先学什么",
            "roadmap",
            "plan",
        ],
    )

    for keyword in summary_keywords:
        if keyword in lowered:
            return "summary"
    
    for keyword in study_plan_keywords:
        if keyword in lowered:
            return "study_plan"

    return "qa"

def detect_course_anchor(question: str) -> str | None:
    """
    加课程标题 / lesson 锚点感知。

    目标：
        如果问题里已经包含课程模块名/lesson 名
        就优先生成一个“课程锚点 query”
        让检索更容易命中对应 notebook/lesson
        这比现在只看 tool use / planning / memory / rag 更贴课程语料。
    """
    lowered = question.lower()

    if "04-tool-use" in lowered or "tool use" in lowered:
        return "04 Tool Use 学习摘要"

    if "05-agentic-rag" in lowered or "agentic rag" in lowered:
        return "05 Agentic RAG 学习摘要"

    if "07-planning-design" in lowered or "planning" in lowered:
        return "07 Planning Design 学习摘要"

    if "08-multi-agent" in lowered or "multi agent" in lowered:
        return "08 Multi Agent 学习摘要"

    if "09-metacognition" in lowered or "metacognition" in lowered:
        return "09 Metacognition 学习摘要"

    if "13-agent-memory" in lowered or "agent memory" in lowered:
        return "13 Agent Memory 学习摘要"

    if "lesson 1" in lowered:
        return "Lesson 1 学习摘要"

    if "lesson 2" in lowered:
        return "Lesson 2 学习摘要"

    return None

def dedupe_queries(queries: list[str]) -> list[str]: # 去重逻辑
    seen: set[str] = set()
    deduped: list[str] = []

    for item in queries:
        text = item.strip()
        if not text or text in seen:
            continue
        seen.add(text)
        deduped.append(text)

    return deduped

def build_primary_queries(
    question: str,
    task_type: str,
    memory: dict | None = None,
) -> list[str]:
    """原始问题 + 任务类型基础改写"""
    queries: list[str] = [question.strip()]
    lowered = question.lower()

    if task_type == "qa":
        if "tool use" in lowered:
            queries.append("tool use agent tool calling")
        elif "agentic rag" in lowered:
            queries.append("agentic rag retrieval augmented generation agent")
        elif "planning" in lowered:
            queries.append("planning agent task decomposition reasoning")
        elif "memory" in lowered:
            queries.append("agent memory short term long term memory")

    elif task_type == "summary":
        queries.append(f"{question.strip()} notebook lesson 总结")

    elif task_type == "study_plan":
        goal = (memory or {}).get("learning_goal", "").strip()
        if goal:
            queries.append(f"{goal} 学习顺序 学习路线")
        else:
            queries.append(f"{question.strip()} 学习顺序 lesson roadmap")

    return dedupe_queries(queries)

def build_anchor_queries(question: str) -> list[str]:
    """课程标题 / lesson 锚点"""
    course_anchor = detect_course_anchor(question)
    if not course_anchor:
        return []
    return [course_anchor]

def build_memory_queries(
    task_type: str,
    memory: dict | None = None,
) -> list[str]:
    """goal / scope / recent_focus"""
    memory = memory or {}
    goal = memory.get("learning_goal", "").strip()
    scope = memory.get("preferred_scope", "").strip()
    recent_focus = memory.get("recent_focus", "").strip()

    queries: list[str] = []

    if task_type == "study_plan":
        if goal:
            queries.append(f"{goal} 学习顺序 学习路线")
        if scope:
            queries.append(scope)
        if recent_focus:
            queries.append(f"{recent_focus} 学习顺序 学习路线")

    elif task_type == "summary":
        if recent_focus:
            queries.append(f"{recent_focus} lesson notebook 总结")

    else:
        if recent_focus:
            queries.append(recent_focus)

    return dedupe_queries(queries)


def build_retrieval_queries(
        question: str,
        task_type: str,
        memory: dict | None = None
) -> list[str]: # 纯组装器
    """
    第一轮：正常检索
    为一次检索构造原始 query 和少量增强 query。
    """
    queries: list[str] = []
    queries.extend(build_primary_queries(question, task_type, memory=memory))
    queries.extend(build_anchor_queries(question))
    queries.extend(build_memory_queries(task_type, memory=memory))
    return dedupe_queries(queries)

def build_query_rewrite_prompt( # 小 prompt builder
    question: str,
    task_type: str,
    memory: dict | None = None,
) -> str:
    memory = memory or {}
    goal = memory.get("learning_goal", "").strip()
    scope = memory.get("preferred_scope", "").strip()
    recent_focus = memory.get("recent_focus", "").strip()

    return f"""你要为课程检索系统生成 1 条更适合检索的 query。

用户原问题：
{question}

任务类型：
{task_type}

用户学习目标：
{goal or "无"}

用户学习范围：
{scope or "无"}

最近学习重点：
{recent_focus or "无"}

要求：
1. 只输出 1 行 query，不要解释。
2. query 要更适合课程资料检索。
3. 尽量保留课程主题词，如 tool use、agentic rag、planning、lesson、notebook。
4. 如果是 study_plan，可加入“学习顺序 / 学习路线 / roadmap / lesson”等词。
5. 不要输出 JSON，不要加编号，不要加引号。
"""

def build_llm_retry_query( # LLM rewrite 函数
    question: str,
    task_type: str,
    settings,
    client,
    memory: dict | None = None,
) -> str | None:
    prompt = build_query_rewrite_prompt(
        question,
        task_type,
        memory=memory,
    )

    response = client.chat.completions.create(
        model=settings.model_name,
        messages=[
            {"role": "system", "content": "你是一个课程检索 query rewrite 助手。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )

    content = response.choices[0].message.content.strip()
    if not content:
        return None

    first_line = content.splitlines()[0].strip()
    if not first_line:
        return None

    if first_line.startswith("{") or first_line.startswith("["):
        return None

    if any(token in first_line for token in ['"answer"', '"suggestions"', '"sources"']): # 把“模型回答 JSON”过滤掉，不要当 rewrite query
        """
        这样模型一旦返回：
            JSON
            列表
            结构化回答字段
            
        就不会被当成 rewrite query。
        """
        return None

    return first_line

def build_retry_retrieval_queries( # 触发了retry
        question: str,
        task_type: str,
        memory: dict | None = None,
        settings=None,
        client=None,
) -> list[str]:
    """
    第二轮更强 query
    第二轮：带更多 goal/scope/recent_focus 的补强检索
    """
    queries: list[str] = []

    if task_type == "summary":
        queries.append(f"{question.strip()} notebook lesson summary")
    elif task_type == "study_plan":
        queries.append(f"{question.strip()} 学习顺序 roadmap lesson")
    else:
        queries.append(f"{question.strip()} agent course concept")

    queries.extend(build_anchor_queries(question))
    queries.extend(build_memory_queries(task_type, memory=memory))

    
    if settings is not None and client is not None:
        try:
            llm_query = build_llm_retry_query(# 多生成 1 条 LLM 补强 query
                question,
                task_type,
                settings=settings,
                client=client,
                memory=memory,
            )
        except Exception:
            llm_query = None
        if llm_query:
            queries.append(llm_query)

    return dedupe_queries(queries)


def update_completed_topic(memory: dict, topic_title: str) -> None:
    # 如果 memory 里已经有 completed_topics，就拿出来, 如果没有，就先创建一个空列表
    completed_topic = memory.setdefault("completed_topics", [])

    if topic_title not in completed_topic:
        completed_topic.append(topic_title)
        
# 最终 sources 输出开始体现 chunk_id 了，之前是纯文档级的 source 路径，现在可以更细粒度地指向具体 chunk了
def format_source_reference(chunk: RetrievedChunk) -> str:
    if chunk.chunk_id:
        return f"{chunk.source}#{chunk.chunk_id}"
    return chunk.source

def is_rag_study_plan_question(question: str) -> bool:
    """判断当前学习路线问题是否明确在问 RAG 到 Agentic RAG 的路径。"""
    lowered = question.lower()
    config = load_study_plan_order_config()
    route_triggers = config.get("route_triggers", {})
    rag_keywords = route_triggers.get("rag_route_keywords", ["rag"])
    return any(keyword.lower() in lowered for keyword in rag_keywords)

def post_rank_study_plan_results(
    question: str,
    retrieved_chunks: list[RetrievedChunk],
) -> list[RetrievedChunk]:
    """
    对 study_plan 场景做最终后排序。

    目标：
    1. 保留 reranker 已经召回出的候选
    2. 但最终顺序更贴近课程学习主线
    3. 如果是 RAG 路线问题，则优先把 2-2 的 RAG 课程排到前面
    """
    if not retrieved_chunks:
        return []

    config = load_study_plan_order_config()
    default_title_order = config.get("default_title_order", [])
    rag_route_priorities = config.get("rag_route_priorities", [])

    title_order_map = {
        title: index for index, title in enumerate(default_title_order)
    }

    def get_rag_priority(item: RetrievedChunk) -> int:
        source = item.source

        if is_rag_study_plan_question(question):# 如果问题里明确提到了 RAG，那我们就按照配置里针对 RAG 路线的优先级来排序，把 2-2 的 RAG 课程放在更前面，确保满足用户的查询意图。
            for index, rule in enumerate(rag_route_priorities):
                rule_type = rule.get("type")
                rule_value = rule.get("value", "")

                if rule_type == "source_contains" and rule_value in item.source:
                    return index

                if rule_type == "title_equals" and item.title == rule_value:
                    return index

        return 999

    def get_default_study_plan_priority(item: RetrievedChunk) -> int:
        # 普通学习路线优先遵守课程主线顺序，未知标题统一放后。
        return title_order_map.get(item.title, 999)

    return sorted(
        retrieved_chunks,
        key=lambda item: (
            get_rag_priority(item),
            get_default_study_plan_priority(item),
            -item.score,
        ),
    )
