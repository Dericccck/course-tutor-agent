# agent.py：调模型
# 串主流程：读取问题、调检索、组装上下文、调模型、返回结构化结果
import json

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
from retriever import retrieve_documents
from schemas import AgentAnswer, Document

def build_client(settings: Settings) -> OpenAI:
    client_kwargs = {"api_key": settings.api_key}

    if settings.base_url:
        client_kwargs["base_url"] = settings.base_url

    return OpenAI(**client_kwargs)

def ask_course_agent(question: str, documents: list[Document], settings: Settings | None = None) -> AgentAnswer:
    active_settings = settings or get_settings()
    validate_settings(active_settings)

    retrieved_chunks = retrieve_documents(query=question, documents=documents, top_k=active_settings.retrieval_top_k)

    if not retrieved_chunks:
        return AgentAnswer(
            answer="当前没有检索到相关课程资料，暂时无法回答这个问题",
            suggestions=["换一个更具体的问题试试","优先使用课程名称、章节名或关键词提问"],
            sources=[]
        )
    
    client = build_client(active_settings)
    task_type = detect_task_type(question)
    if task_type == "summary":
        user_prompt = build_summary_prompt(question, retrieved_chunks)
    elif task_type == "study_plan":
        user_prompt = build_study_plan_prompt(question, retrieved_chunks)
    else:
        user_prompt = build_user_prompt(question, retrieved_chunks)

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
            sources=[chunk.source for chunk in retrieved_chunks],
        )
    
    if not answer.sources:
        answer.sources = [chunk.source for chunk in retrieved_chunks]

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
