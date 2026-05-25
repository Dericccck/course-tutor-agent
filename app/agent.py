# agent.py：调模型
# 串主流程：读取问题、调检索、组装上下文、调模型、返回结构化结果
import json
from vector_store import VectorStore

# pyrefly: ignore [missing-import]
from openai import OpenAI
# pyrefly: ignore [missing-import]
from pydantic import ValidationError

from config import Settings, get_settings, validate_settings
from prompts import (
    SYSTEM_PROMPT,
    build_user_prompt,
    build_summary_prompt,
    build_study_plan_prompt,
)
from retriever import retrieve_documents, retrieve_chunks
from schemas import AgentAnswer, Document, DocumentChunk, RetrievedChunk

def build_client(settings: Settings) -> OpenAI:
    client_kwargs = {"api_key": settings.api_key}

    if settings.base_url:
        client_kwargs["base_url"] = settings.base_url

    return OpenAI(**client_kwargs)

# 混合检索 用 retrieve_chunks(...) 取一份结果，用 vector_store.search(...) 再取一份结果，然后把两份结果合并去重，最终返回 top_k 条结果。这样做的好处是：
# 1. retrieve_chunks(...) 的结果通常更精准，因为它直接基于文本内容进行匹配，能够捕捉到一些细粒度的相关信息；而 vector_store.search(...) 的结果可能更全面，因为它基于向量表示进行匹配。通过混合检索，我们可以兼顾精准性和全面性，提升整体的检索效果。
# 2. retrieve_chunks(...) 的结果可以作为 vector_store.search(...) 的补充，当 retrieve_chunks(...) 没有检索到足够的相关内容时，vector_store.search(...) 可以提供更多的候选项，增加找到相关信息的机会。反过来，当 retrieve_chunks(...) 已经检索到足够的相关内容时，我们也可以通过混合检索来引入一些 vector_store.search(...) 的结果，增加多样性和覆盖面。
def merge_retrieval_results(primary: list[RetrievedChunk], secondary: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
    merged: list[RetrievedChunk] = []
    seen: set[tuple[str, str | None]] = set()  # 用于去重，记录已经添加过的 (source, chunk_id) 组合

    for item in primary + secondary:
        key = (item.source, item.chunk_id)
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)

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
    ) -> AgentAnswer:
    active_settings = settings or get_settings()
    validate_settings(active_settings)

    if active_settings.retrieval_mode == "hybrid": # 混合检索
        if chunks is None:
            raise ValueError("Chunks data must be provided when RETRIEVAL_MODE=hybrid.")
        if vector_store is None:
            raise ValueError("Vector store must be provided when RETRIEVAL_MODE=hybrid.")
        # 这里先让 lexical_results 放前面，是因为：标题/关键词精确命中对课程问答很重要    vector 先作为补充召回
        lexical_results = retrieve_chunks(
            query=question,
            chunks=chunks,
            top_k=active_settings.retrieval_top_k,
        )
        vector_results = vector_store.search(
            query=question,
            top_k=active_settings.retrieval_top_k,
        )
        retrieved_chunks = merge_retrieval_results(lexical_results, vector_results, top_k=active_settings.retrieval_top_k)
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
