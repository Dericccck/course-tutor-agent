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

def build_client(settings: Settings) -> OpenAI:
    client_kwargs = {"api_key": settings.api_key}

    if settings.base_url:
        client_kwargs["base_url"] = settings.base_url

    return OpenAI(**client_kwargs)

def load_study_plan_order_config() -> dict:
    """读取 study_plan 排序配置。"""
    with STUDY_PLAN_ORDER_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)

# 这个函数的作用是从给定的 source 路径中提取课程组信息。我们假设课程组的信息包含在路径的某个部分，并且以 "1-" 或 "2-" 开头。通过这个函数，我们可以在后续的检索结果排序中优先展示与用户问题相关的课程组内容，从而提升用户体验和检索结果的相关性。
def get_course_group(source: str) -> str | None:
    parts = Path(source).parts
    for part in parts:
        if part.startswith(("1-", "2-")):
            return part
    return None

# 这个函数的作用是对检索到的结果进行过滤和排序，优先保留与首条结果来自同一课程组的内容。这样做的好处是可以提升检索结果的相关性和一致性，尤其是在用户提问中包含了特定课程组信息的情况下。通过这种方式，我们可以更好地满足用户的查询意图，提供更精准和有针对性的回答。
def narrow_summary_results(retrieved_chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    if not retrieved_chunks:
        return []

    target_source = retrieved_chunks[0].source
    same_source = [
        item for item in retrieved_chunks
        if item.source == target_source
    ]
    return same_source or retrieved_chunks

# 混合检索 用 retrieve_chunks(...) 取一份结果，用 vector_store.search(...) 再取一份结果，然后把两份结果合并去重，最终返回 top_k 条结果。这样做的好处是：
# 1. retrieve_chunks(...) 的结果通常更精准，因为它直接基于文本内容进行匹配，能够捕捉到一些细粒度的相关信息；而 vector_store.search(...) 的结果可能更全面，因为它基于向量表示进行匹配。通过混合检索，我们可以兼顾精准性和全面性，提升整体的检索效果。
# 2. retrieve_chunks(...) 的结果可以作为 vector_store.search(...) 的补充，当 retrieve_chunks(...) 没有检索到足够的相关内容时，vector_store.search(...) 可以提供更多的候选项，增加找到相关信息的机会。反过来，当 retrieve_chunks(...) 已经检索到足够的相关内容时，我们也可以通过混合检索来引入一些 vector_store.search(...) 的结果，增加多样性和覆盖面。
def merge_retrieval_results(
    primary: list[RetrievedChunk],
    secondary: list[RetrievedChunk],
    top_k: int,
    max_per_source: int = 2,
) -> list[RetrievedChunk]:
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

    candidates = primary + same_group_secondary + other_group_secondary

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

    if active_settings.retrieval_mode == "hybrid": # 混合检索
        if chunks is None:
            raise ValueError("Chunks data must be provided when RETRIEVAL_MODE=hybrid.")
        if vector_store is None:
            raise ValueError("Vector store must be provided when RETRIEVAL_MODE=hybrid.")
        candidate_top_k = max(active_settings.retrieval_top_k * 3, 10)
        # 这里先让 lexical_results 放前面，是因为：标题/关键词精确命中对课程问答很重要    vector 先作为补充召回
        lexical_results = retrieve_chunks(
            query=question,
            chunks=chunks,
            top_k=candidate_top_k,
        )
        vector_results = vector_store.search(
            query=question,
            top_k=candidate_top_k,
        )
        retrieved_chunks = merge_retrieval_results(lexical_results, vector_results, top_k=active_settings.retrieval_top_k) # lexical 定主轴，vector 做补充
        if reranker is not None: # 如果提供了 reranker，就对混合检索的结果进行重新排序，进一步提升相关性。这里我们把混合检索的结果作为 reranker 的输入，让它根据查询和每条结果的内容来打分排序，从而把最相关的结果排在前面，提升最终返回给用户的答案的质量和准确性。
            retrieved_chunks = reranker.rerank(
                question,
                retrieved_chunks,
                top_k=active_settings.retrieval_top_k,
            )
    elif active_settings.retrieval_mode == "vector": # 目前向量检索还没做，所以先直接报错，等后续完善了再放开这个选项
        if vector_store is None:
            raise ValueError("Vector store must be provided when RETRIEVAL_MODE=vector.")
        retrieved_chunks = vector_store.search(
            query=question, 
            top_k=active_settings.retrieval_top_k,
        )
    elif active_settings.retrieval_mode == "chunk" and chunks is not None: # 优先使用切分后的 chunk 进行检索，只有在没有提供 chunks 或者 retrieval_mode 设置为 document 时才退回到文档级检索
        retrieved_chunks = retrieve_chunks(
            query=question,
            chunks=chunks,
            top_k=active_settings.retrieval_top_k,
        )
    else: # 否则就退回到最原始的文档级检索（虽然效率更低，但至少能工作）
        retrieved_chunks = retrieve_documents(
            query=question,
            documents=documents,
            top_k=active_settings.retrieval_top_k,
        )

    if not retrieved_chunks:
        return AgentAnswer(
            answer="当前没有检索到相关课程资料，暂时无法回答这个问题",
            suggestions=["换一个更具体的问题试试","优先使用课程名称、章节名或关键词提问"],
            sources=[]
        )
    
    client = build_client(active_settings)
    task_type = detect_task_type(question)
    if task_type == "summary":
        retrieved_chunks = narrow_summary_results(retrieved_chunks)
    if task_type == "study_plan":
        retrieved_chunks = post_rank_study_plan_results(question, retrieved_chunks)
    if task_type == "summary":
        user_prompt = build_summary_prompt(question, retrieved_chunks, memory=memory)
    elif task_type == "study_plan":
        user_prompt = build_study_plan_prompt(question, retrieved_chunks, memory=memory)
    else:
        user_prompt = build_user_prompt(question, retrieved_chunks, memory=memory)

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
            sources=[format_source_reference(chunk) for chunk in retrieved_chunks],
        )
    
    answer.sources = [format_source_reference(chunk) for chunk in retrieved_chunks]

    return answer


def detect_task_type(question: str) -> str:
    lowered = question.lower()

    summary_keywords = [
        "总结",
        "概述",
        "概要",
        "讲什么",
        "这一节",
        "这节课",
        "notebook",
        "lesson",
    ]

    study_plan_keywords = [
        "学习顺序",
        "学习路线",
        "学习计划",
        "怎么学",
        "从哪里开始",
        "先学什么",
        "roadmap",
        "plan",
    ]

    for keyword in summary_keywords:
        if keyword in lowered:
            return "summary"
    
    for keyword in study_plan_keywords:
        if keyword in lowered:
            return "study_plan"

    return "qa"


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
    return "rag" in lowered

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
