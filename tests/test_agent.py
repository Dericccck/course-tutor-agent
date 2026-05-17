# 测试 agent 主流程中的本地逻辑：
# 1. 没有检索结果时，应返回兜底回答
# 2. summary 任务应自动更新 completed_topics
# 3. 模型返回非法 JSON 时，应走 fallback 而不是报错

from types import SimpleNamespace

import agent
from schemas import Document, DocumentChunk, RetrievedChunk


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


def make_document_chunk(title: str = "Test Chunk") -> DocumentChunk:
    # 构造一个最小可用的切块对象，供 chunk 检索路径测试使用
    return DocumentChunk(
        source="/tmp/test.md",
        title=title,
        chunk_id=f"{title}-chunk-1",
        content=f"{title} 的 chunk 内容",
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


def test_ask_course_agent_prefers_chunk_retrieval_when_chunks_provided(monkeypatch):
    # 传入 chunks 时，应优先走 chunk 检索，而不是退回 document 检索
    monkeypatch.setattr(agent, "validate_settings", lambda settings: None)

    called = {"chunk": False, "document": False}

    def fake_retrieve_chunks(query, chunks, top_k):
        called["chunk"] = True
        return [make_chunk("04 Tool Use 学习摘要")]

    def fake_retrieve_documents(query, documents, top_k):
        called["document"] = True
        return [make_chunk("不会被使用")]

    monkeypatch.setattr(agent, "retrieve_chunks", fake_retrieve_chunks)
    monkeypatch.setattr(agent, "retrieve_documents", fake_retrieve_documents)

    fake_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content='{"answer": "chunk 检索结果", "suggestions": [], "sources": []}'
                )
            )
        ]
    )

    class FakeCompletions:
        def create(self, **kwargs):
            return fake_response

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    monkeypatch.setattr(agent, "build_client", lambda settings: fake_client)

    result = agent.ask_course_agent(
        question="tool use 是什么？",
        documents=[make_document()],
        settings=make_settings(),
        memory={},
        chunks=[make_document_chunk("04 Tool Use 学习摘要")],
    )

    assert called["chunk"] is True
    assert called["document"] is False
    assert result.answer == "chunk 检索结果"


def test_ask_course_agent_updates_completed_topics_for_summary_with_chunks(monkeypatch):
    # summary 场景在 chunk 检索路径下，也应把第一条标题写入 completed_topics
    monkeypatch.setattr(agent, "validate_settings", lambda settings: None)
    monkeypatch.setattr(
        agent,
        "retrieve_chunks",
        lambda query, chunks, top_k: [make_chunk("07 Planning Design 学习摘要")],
    )
    monkeypatch.setattr(
        agent,
        "retrieve_documents",
        lambda query, documents, top_k: [make_chunk("不会被使用")],
    )

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

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
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
        chunks=[make_document_chunk("07 Planning Design 学习摘要")],
    )

    assert "07 Planning Design 学习摘要" in memory["completed_topics"]


def test_format_source_reference_includes_chunk_id_when_present():
    # 有 chunk_id 时，应拼成 source#chunk_id 的细粒度引用
    chunk = make_chunk("04 Tool Use 学习摘要")
    chunk.chunk_id = "04 Tool Use 学习摘要-chunk-1"

    result = agent.format_source_reference(chunk)

    assert result == "/tmp/04 Tool Use 学习摘要.md#04 Tool Use 学习摘要-chunk-1"


def test_format_source_reference_returns_source_when_chunk_id_missing():
    # 没有 chunk_id 时，应退回文档级路径
    chunk = make_chunk("04 Tool Use 学习摘要")
    chunk.chunk_id = None

    result = agent.format_source_reference(chunk)

    assert result == "/tmp/04 Tool Use 学习摘要.md"


def test_ask_course_agent_overrides_model_sources_with_chunk_references(monkeypatch):
    # 即使模型自己返回了 sources，也应以系统生成的 chunk 级引用为准
    monkeypatch.setattr(agent, "validate_settings", lambda settings: None)

    retrieved = [
        RetrievedChunk(
            source="/tmp/04-tool-use.md",
            title="04 Tool Use 学习摘要",
            chunk_id="04 Tool Use 学习摘要-chunk-1",
            snippet="tool use 相关片段",
            score=10.0,
            tags=["agent"],
        )
    ]

    monkeypatch.setattr(
        agent,
        "retrieve_chunks",
        lambda query, chunks, top_k: retrieved,
    )
    monkeypatch.setattr(
        agent,
        "retrieve_documents",
        lambda query, documents, top_k: [make_chunk("不会被使用")],
    )

    fake_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content='{"answer": "正常回答", "suggestions": [], "sources": ["/tmp/old-source.md"]}'
                )
            )
        ]
    )

    class FakeCompletions:
        def create(self, **kwargs):
            return fake_response

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    monkeypatch.setattr(agent, "build_client", lambda settings: fake_client)

    result = agent.ask_course_agent(
        question="tool use 是什么？",
        documents=[make_document()],
        settings=make_settings(),
        memory={},
        chunks=[make_document_chunk("04 Tool Use 学习摘要")],
    )

    assert result.answer == "正常回答"
    assert result.sources == ["/tmp/04-tool-use.md#04 Tool Use 学习摘要-chunk-1"]
