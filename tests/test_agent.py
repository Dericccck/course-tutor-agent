# 测试 agent 主流程中的本地逻辑：
# 1. 没有检索结果时，应返回兜底回答
# 2. summary 任务应自动更新 completed_topics
# 3. 模型返回非法 JSON 时，应走 fallback 而不是报错

from types import SimpleNamespace

import agent
from schemas import Document, RetrievedChunk


def make_document(title: str = "Test Title") -> Document:
    # 构造一个最小可用的文档对象，供 agent 测试使用
    return Document(
        source="/tmp/test.md",
        title=title,
        content="test content",
        doc_type="md",
        tags=["agent"],
    )


def make_chunk(title: str) -> RetrievedChunk:
    # 构造一个最小可用的检索结果对象
    return RetrievedChunk(
        source=f"/tmp/{title}.md",
        title=title,
        snippet=f"{title} 的相关片段",
        score=10.0,
        tags=["agent"],
    )


def make_settings() -> SimpleNamespace:
    # 用一个简单对象模拟 Settings，避免依赖真实环境变量
    return SimpleNamespace(
        llm_provider="github",
        model_name="gpt-4.1-mini",
        api_key="fake-key",
        base_url="https://models.inference.ai.azure.com/",
        course_source_root="/tmp",
        retrieval_top_k=5,
    )


def test_ask_course_agent_returns_fallback_when_no_retrieved_chunks(monkeypatch):
    # 当检索结果为空时，应直接返回兜底回答，而不是继续调模型
    monkeypatch.setattr(agent, "validate_settings", lambda settings: None)
    monkeypatch.setattr(agent, "retrieve_documents", lambda query, documents, top_k: [])

    result = agent.ask_course_agent(
        question="一个完全找不到资料的问题",
        documents=[make_document()],
        settings=make_settings(),
        memory={},
    )

    assert "当前没有检索到相关课程资料" in result.answer
    assert result.sources == []


def test_ask_course_agent_updates_completed_topics_for_summary(monkeypatch):
    # summary 任务在命中资料时，应把第一条标题写入 completed_topics
    monkeypatch.setattr(agent, "validate_settings", lambda settings: None)
    monkeypatch.setattr(
        agent,
        "retrieve_documents",
        lambda query, documents, top_k: [make_chunk("07 Planning Design 学习摘要")],
    )
    monkeypatch.setattr(agent, "build_client", lambda settings: SimpleNamespace())

    fake_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content='{"answer": "总结内容", "suggestions": [], "sources": []}'
                )
            )
        ]
    )

    class FakeCompletions:
        def create(self, **kwargs):
            return fake_response

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=FakeCompletions())
    )
    monkeypatch.setattr(agent, "build_client", lambda settings: fake_client)

    memory = {
        "learning_goal": "",
        "preferred_scope": "",
        "completed_topics": [],
    }

    agent.ask_course_agent(
        question="帮我总结 07-planning-design 这一节在讲什么",
        documents=[make_document()],
        settings=make_settings(),
        memory=memory,
    )

    assert "07 Planning Design 学习摘要" in memory["completed_topics"]


def test_ask_course_agent_falls_back_when_model_returns_invalid_json(monkeypatch):
    # 模型如果返回非法 JSON，不应抛异常，而应走 fallback
    monkeypatch.setattr(agent, "validate_settings", lambda settings: None)
    monkeypatch.setattr(
        agent,
        "retrieve_documents",
        lambda query, documents, top_k: [make_chunk("04 Tool Use 学习摘要")],
    )

    fake_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="这不是合法 JSON"
                )
            )
        ]
    )

    class FakeCompletions:
        def create(self, **kwargs):
            return fake_response

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=FakeCompletions())
    )
    monkeypatch.setattr(agent, "build_client", lambda settings: fake_client)

    result = agent.ask_course_agent(
        question="tool use 是什么？",
        documents=[make_document()],
        settings=make_settings(),
        memory={},
    )

    assert result.answer == "这不是合法 JSON"
    assert "/tmp/04 Tool Use 学习摘要.md" in result.sources
