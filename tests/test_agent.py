# 测试 agent 主流程中的本地逻辑：
# 1. 没有检索结果时，应返回兜底回答
# 2. summary 任务应自动更新 completed_topics
# 3. 模型返回非法 JSON 时，应走 fallback 而不是报错

import json
from types import SimpleNamespace

import agent
from schemas import Document, DocumentChunk, RetrievedChunk
from reranker import FakeReranker


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


def make_settings(
    retrieval_mode: str = "chunk",
    hybrid_candidate_multiplier: int = 3,
    hybrid_candidate_minimum: int = 10,
) -> SimpleNamespace:
    # 用一个简单对象模拟 Settings，避免依赖真实环境变量
    return SimpleNamespace(
        llm_provider="github",
        model_name="gpt-4.1-mini",
        api_key="fake-key",
        base_url="https://models.inference.ai.azure.com/",
        course_source_root="/tmp",
        retrieval_top_k=5,
        hybrid_candidate_multiplier=hybrid_candidate_multiplier,
        hybrid_candidate_minimum=hybrid_candidate_minimum,
        retrieval_mode=retrieval_mode,
        embedding_provider="hash",
        embedding_model_name="BAAI/bge-m3",
        embedding_cache_dir=None,
        course_include_dirs=["2-3-ai-agents-for-beginners"],
        reranker_provider="none",
        reranker_model_name="BAAI/bge-reranker-base",
        reranker_cache_dir=None,
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


def test_ask_course_agent_uses_document_retrieval_when_mode_is_document(monkeypatch):
    # 即使传入了 chunks，只要 retrieval_mode=document，也应走文档级检索
    monkeypatch.setattr(agent, "validate_settings", lambda settings: None)

    called = {"chunk": False, "document": False}

    def fake_retrieve_chunks(query, chunks, top_k):
        called["chunk"] = True
        return [make_chunk("不会被使用")]

    def fake_retrieve_documents(query, documents, top_k):
        called["document"] = True
        return [make_chunk("04 Tool Use 学习摘要")]

    monkeypatch.setattr(agent, "retrieve_chunks", fake_retrieve_chunks)
    monkeypatch.setattr(agent, "retrieve_documents", fake_retrieve_documents)

    fake_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content='{"answer": "document 检索结果", "suggestions": [], "sources": []}'
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
        settings=make_settings(retrieval_mode="document"),
        memory={},
        chunks=[make_document_chunk("04 Tool Use 学习摘要")],
    )

    assert called["document"] is True
    assert called["chunk"] is False
    assert result.answer == "document 检索结果"


def test_ask_course_agent_raises_for_vector_retrieval_mode(monkeypatch):
    # vector 模式下如果没有传入 vector_store，应明确报错
    monkeypatch.setattr(agent, "validate_settings", lambda settings: None)

    try:
        agent.ask_course_agent(
            question="tool use 是什么？",
            documents=[make_document()],
            settings=make_settings(retrieval_mode="vector"),
            memory={},
            chunks=[make_document_chunk("04 Tool Use 学习摘要")],
            vector_store=None,  # 即使不传 vector_store，也应优先抛出未实现异常，而不是因为缺参数而抛 ValueError
        )
        assert False, "ask_course_agent should raise ValueError when vector_store is missing"
    except ValueError as exc:
        assert "Vector store must be provided" in str(exc)


def test_ask_course_agent_uses_vector_store_when_mode_is_vector(monkeypatch):
    # vector 模式下，如果传入 vector_store，应使用它的 search 结果继续主流程
    monkeypatch.setattr(agent, "validate_settings", lambda settings: None)

    called = {"document": False, "chunk": False, "vector": False}

    def fake_retrieve_documents(query, documents, top_k):
        called["document"] = True
        return [make_chunk("不会被使用")]

    def fake_retrieve_chunks(query, chunks, top_k):
        called["chunk"] = True
        return [make_chunk("不会被使用")]

    class FakeVectorStore:
        def search(self, query, top_k=5):
            called["vector"] = True
            return [
                RetrievedChunk(
                    source="/tmp/vector-source.md",
                    title="Vector 命中结果",
                    chunk_id="vector-chunk-1",
                    snippet="向量检索命中的片段",
                    score=9.5,
                    tags=["agent"],
                )
            ]

    monkeypatch.setattr(agent, "retrieve_documents", fake_retrieve_documents)
    monkeypatch.setattr(agent, "retrieve_chunks", fake_retrieve_chunks)

    fake_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content='{"answer": "vector 检索结果", "suggestions": [], "sources": []}'
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
        settings=make_settings(retrieval_mode="vector"),
        memory={},
        chunks=[make_document_chunk("04 Tool Use 学习摘要")],
        vector_store=FakeVectorStore(),
    )

    assert called["vector"] is True
    assert called["document"] is False
    assert called["chunk"] is False
    assert result.answer == "vector 检索结果"
    assert result.sources == ["/tmp/vector-source.md#vector-chunk-1"]


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


def test_merge_retrieval_results_deduplicates_and_preserves_priority():
    primary = [
        RetrievedChunk(
            source="/tmp/tool.md",
            title="04 Tool Use 学习摘要",
            chunk_id="tool-1",
            snippet="primary",
            score=10.0,
            tags=["agent"],
        ),
        RetrievedChunk(
            source="/tmp/framework.md",
            title="02 Explore Agentic Frameworks 学习摘要",
            chunk_id="framework-1",
            snippet="primary-2",
            score=9.0,
            tags=["agent"],
        ),
    ]
    secondary = [
        RetrievedChunk(
            source="/tmp/tool.md",
            title="04 Tool Use 学习摘要",
            chunk_id="tool-1",
            snippet="duplicate",
            score=8.5,
            tags=["agent"],
        ),
        RetrievedChunk(
            source="/tmp/rag.md",
            title="05 Agentic RAG 学习摘要",
            chunk_id="rag-1",
            snippet="secondary",
            score=8.0,
            tags=["rag"],
        ),
    ]

    merged = agent.merge_retrieval_results(primary, secondary, top_k=3)

    assert len(merged) == 3
    assert merged[0].chunk_id == "tool-1"
    assert merged[1].chunk_id == "framework-1"
    assert merged[2].chunk_id == "rag-1"


def test_ask_course_agent_uses_hybrid_retrieval_when_mode_is_hybrid(monkeypatch):
    monkeypatch.setattr(agent, "validate_settings", lambda settings: None)

    called = {"chunk": False, "document": False, "vector": False}

    def fake_retrieve_chunks(query, chunks, top_k):
        called["chunk"] = True
        return [
            RetrievedChunk(
                source="/tmp/tool.md",
                title="04 Tool Use 学习摘要",
                chunk_id="tool-1",
                snippet="lexical result",
                score=10.0,
                tags=["agent"],
            )
        ]

    def fake_retrieve_documents(query, documents, top_k):
        called["document"] = True
        return [make_chunk("不会被使用")]

    class FakeVectorStore:
        def search(self, query, top_k=5):
            called["vector"] = True
            return [
                RetrievedChunk(
                    source="/tmp/tool.md",
                    title="04 Tool Use 学习摘要",
                    chunk_id="tool-1",
                    snippet="duplicate vector result",
                    score=9.5,
                    tags=["agent"],
                ),
                RetrievedChunk(
                    source="/tmp/framework.md",
                    title="02 Explore Agentic Frameworks 学习摘要",
                    chunk_id="framework-1",
                    snippet="vector result",
                    score=8.5,
                    tags=["agent"],
                ),
            ]

    monkeypatch.setattr(agent, "retrieve_chunks", fake_retrieve_chunks)
    monkeypatch.setattr(agent, "retrieve_documents", fake_retrieve_documents)

    fake_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content='{"answer": "hybrid 检索结果", "suggestions": [], "sources": []}'
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
        settings=make_settings(retrieval_mode="hybrid"),
        memory={},
        chunks=[make_document_chunk("04 Tool Use 学习摘要")],
        vector_store=FakeVectorStore(),
    )

    assert called["chunk"] is True
    assert called["vector"] is True
    assert called["document"] is False
    assert result.answer == "hybrid 检索结果"
    assert result.sources == [
        "/tmp/tool.md#tool-1",
        "/tmp/framework.md#framework-1",
    ]


def test_ask_course_agent_raises_when_hybrid_mode_missing_vector_store(monkeypatch):
    monkeypatch.setattr(agent, "validate_settings", lambda settings: None)

    try:
        agent.ask_course_agent(
            question="tool use 是什么？",
            documents=[make_document()],
            settings=make_settings(retrieval_mode="hybrid"),
            memory={},
            chunks=[make_document_chunk("04 Tool Use 学习摘要")],
            vector_store=None,
        )
        assert False, "ask_course_agent should raise ValueError when hybrid vector_store is missing"
    except ValueError as exc:
        assert "Vector store must be provided" in str(exc)


def test_narrow_summary_results_keeps_only_first_source():
    retrieved_chunks = [
        RetrievedChunk(
            source="/tmp/07-planning-design.md",
            title="07 Planning Design 学习摘要",
            chunk_id="07-1",
            snippet="target-1",
            score=10.0,
            tags=["agent"],
        ),
        RetrievedChunk(
            source="/tmp/07-planning-design.md",
            title="07 Planning Design 学习摘要",
            chunk_id="07-2",
            snippet="target-2",
            score=9.0,
            tags=["agent"],
        ),
        RetrievedChunk(
            source="/tmp/other.md",
            title="Other",
            chunk_id="other-1",
            snippet="other",
            score=8.0,
            tags=["rag"],
        ),
    ]

    narrowed = agent.narrow_summary_results(retrieved_chunks)

    assert len(narrowed) == 2
    assert all(item.source == "/tmp/07-planning-design.md" for item in narrowed)


def test_ask_course_agent_narrows_summary_results_to_target_source(monkeypatch):
    monkeypatch.setattr(agent, "validate_settings", lambda settings: None)

    retrieved = [
        RetrievedChunk(
            source="/tmp/07-planning-design.md",
            title="07 Planning Design 学习摘要",
            chunk_id="07-1",
            snippet="target-1",
            score=10.0,
            tags=["agent"],
        ),
        RetrievedChunk(
            source="/tmp/07-planning-design.md",
            title="07 Planning Design 学习摘要",
            chunk_id="07-2",
            snippet="target-2",
            score=9.0,
            tags=["agent"],
        ),
        RetrievedChunk(
            source="/tmp/other.md",
            title="Other",
            chunk_id="other-1",
            snippet="other",
            score=8.0,
            tags=["rag"],
        ),
    ]

    monkeypatch.setattr(
        agent,
        "retrieve_chunks",
        lambda query, chunks, top_k: retrieved,
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

    result = agent.ask_course_agent(
        question="帮我总结 07-planning-design 这一节在讲什么",
        documents=[make_document()],
        settings=make_settings(retrieval_mode="chunk"),
        memory={},
        chunks=[make_document_chunk("07 Planning Design 学习摘要")],
    )

    assert result.sources == [
        "/tmp/07-planning-design.md#07-1",
        "/tmp/07-planning-design.md#07-2",
    ]


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

def test_ask_course_agent_uses_reranker_for_hybrid_results(monkeypatch):
    monkeypatch.setattr(agent, "validate_settings", lambda settings: None)

    def fake_retrieve_chunks(query, chunks, top_k):
        return [
            RetrievedChunk(
                source="/tmp/tool.md",
                title="04 Tool Use 学习摘要",
                chunk_id="tool-1",
                snippet="lexical result",
                score=10.0,
                tags=["agent"],
            )
        ]

    class FakeVectorStore:
        def search(self, query, top_k=5):
            return [
                RetrievedChunk(
                    source="/tmp/framework.md",
                    title="02 Explore Agentic Frameworks 学习摘要",
                    chunk_id="framework-1",
                    snippet="vector result",
                    score=8.5,
                    tags=["agent"],
                )
            ]

    monkeypatch.setattr(agent, "retrieve_chunks", fake_retrieve_chunks)

    fake_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content='{"answer": "hybrid rerank 结果", "suggestions": [], "sources": []}'
                )
            )
        ]
    )

    class FakeCompletions:
        def create(self, **kwargs):
            return fake_response

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    monkeypatch.setattr(agent, "build_client", lambda settings: fake_client)

    reranker = FakeReranker(
        results=[
            RetrievedChunk(
                source="/tmp/framework.md",
                title="02 Explore Agentic Frameworks 学习摘要",
                chunk_id="framework-1",
                snippet="reranked first",
                score=99.0,
                tags=["agent"],
            ),
            RetrievedChunk(
                source="/tmp/tool.md",
                title="04 Tool Use 学习摘要",
                chunk_id="tool-1",
                snippet="reranked second",
                score=88.0,
                tags=["agent"],
            ),
        ]
    )

    result = agent.ask_course_agent(
        question="tool use 是什么？",
        documents=[make_document()],
        settings=make_settings(retrieval_mode="hybrid"),
        memory={},
        chunks=[make_document_chunk("04 Tool Use 学习摘要")],
        vector_store=FakeVectorStore(),
        reranker=reranker,
    )

    assert len(reranker.calls) == 1
    assert result.answer == "hybrid rerank 结果"
    assert result.sources == [
        "/tmp/framework.md#framework-1",
        "/tmp/tool.md#tool-1",
    ]


def test_ask_course_agent_uses_configured_hybrid_candidate_pool(monkeypatch):
    monkeypatch.setattr(agent, "validate_settings", lambda settings: None)
    lexical_top_ks: list[int] = []
    vector_top_ks: list[int] = []

    def fake_retrieve_chunks(query, chunks, top_k):
        lexical_top_ks.append(top_k)
        return [
            RetrievedChunk(
                source="/tmp/tool.md",
                title="04 Tool Use 学习摘要",
                chunk_id="tool-1",
                snippet="lexical result",
                score=10.0,
                tags=["agent"],
            )
        ]

    class FakeVectorStore:
        def search(self, query, top_k=5):
            vector_top_ks.append(top_k)
            return [
                RetrievedChunk(
                    source="/tmp/framework.md",
                    title="02 Explore Agentic Frameworks 学习摘要",
                    chunk_id="framework-1",
                    snippet="vector result",
                    score=8.5,
                    tags=["agent"],
                )
            ]

    monkeypatch.setattr(agent, "retrieve_chunks", fake_retrieve_chunks)

    fake_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content='{"answer": "hybrid 候选池结果", "suggestions": [], "sources": []}'
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
        settings=make_settings(
            retrieval_mode="hybrid",
            hybrid_candidate_multiplier=4,
            hybrid_candidate_minimum=12,
        ),
        memory={},
        chunks=[make_document_chunk("04 Tool Use 学习摘要")],
        vector_store=FakeVectorStore(),
    )

    assert lexical_top_ks == [20]
    assert vector_top_ks == [20]
    assert result.answer == "hybrid 候选池结果"


def test_post_rank_study_plan_results_prefers_default_learning_order():
    retrieved_chunks = [
        RetrievedChunk(
            source="/tmp/protocols.md",
            title="11 Agentic Protocols 学习摘要",
            chunk_id="11-1",
            snippet="protocols",
            score=99.0,
            tags=["agent"],
        ),
        RetrievedChunk(
            source="/tmp/frameworks.md",
            title="02 Explore Agentic Frameworks 学习摘要",
            chunk_id="02-1",
            snippet="frameworks",
            score=80.0,
            tags=["agent"],
        ),
        RetrievedChunk(
            source="/tmp/intro.md",
            title="01 Intro To AI Agents 学习摘要",
            chunk_id="01-1",
            snippet="intro",
            score=70.0,
            tags=["agent"],
        ),
    ]

    ranked = agent.post_rank_study_plan_results(
        "如果我只学 1-* 和 2-*，想做一个 AIAgent 项目，请按课程模块给我安排学习顺序。",
        retrieved_chunks,
    )

    assert [item.title for item in ranked] == [
        "01 Intro To AI Agents 学习摘要",
        "02 Explore Agentic Frameworks 学习摘要",
        "11 Agentic Protocols 学习摘要",
    ]


def test_post_rank_study_plan_results_prefers_rag_curriculum_for_rag_questions():
    retrieved_chunks = [
        RetrievedChunk(
            source="/Users/a1-6/Desktop/AIAgent/code/2-3-ai-agents-for-beginners/05-agentic-rag/notebook-summary.md",
            title="05 Agentic RAG 学习摘要",
            chunk_id="05-1",
            snippet="agentic rag",
            score=100.0,
            tags=["rag"],
        ),
        RetrievedChunk(
            source="/Users/a1-6/Desktop/AIAgent/code/2-2-BuildingAndEvaluatingAdvancedRAGApplications/L2/notebook-summary.md",
            title="Lesson 2 学习摘要",
            chunk_id="L2-1",
            snippet="lesson 2",
            score=70.0,
            tags=["rag"],
        ),
        RetrievedChunk(
            source="/Users/a1-6/Desktop/AIAgent/code/2-2-BuildingAndEvaluatingAdvancedRAGApplications/L1/notebook-summary.md",
            title="Lesson 1 学习摘要",
            chunk_id="L1-1",
            snippet="lesson 1",
            score=60.0,
            tags=["rag"],
        ),
    ]

    ranked = agent.post_rank_study_plan_results(
        "如果我想重点学 RAG，再过渡到 Agentic RAG，应该怎么安排学习顺序？",
        retrieved_chunks,
    )

    assert [item.source for item in ranked] == [
        "/Users/a1-6/Desktop/AIAgent/code/2-2-BuildingAndEvaluatingAdvancedRAGApplications/L1/notebook-summary.md",
        "/Users/a1-6/Desktop/AIAgent/code/2-2-BuildingAndEvaluatingAdvancedRAGApplications/L2/notebook-summary.md",
        "/Users/a1-6/Desktop/AIAgent/code/2-3-ai-agents-for-beginners/05-agentic-rag/notebook-summary.md",
    ]


def test_load_study_plan_order_config_reads_json_file(monkeypatch, tmp_path):
    config_path = tmp_path / "study_plan_order.json"
    config_payload = {
        "default_title_order": ["B", "A"],
        "rag_route_priorities": [
            {"type": "source_contains", "value": "L1"},
        ],
    }
    config_path.write_text(
        json.dumps(config_payload, ensure_ascii=False),
        encoding="utf-8",
    )
    monkeypatch.setattr(agent, "STUDY_PLAN_ORDER_PATH", config_path)

    loaded = agent.load_study_plan_order_config()

    assert loaded == config_payload


def test_post_rank_study_plan_results_uses_loaded_config(monkeypatch):
    monkeypatch.setattr(
        agent,
        "load_study_plan_order_config",
        lambda: {
            "default_title_order": [
                "11 Agentic Protocols 学习摘要",
                "02 Explore Agentic Frameworks 学习摘要",
                "01 Intro To AI Agents 学习摘要",
            ],
            "rag_route_priorities": [],
        },
    )

    retrieved_chunks = [
        RetrievedChunk(
            source="/tmp/intro.md",
            title="01 Intro To AI Agents 学习摘要",
            chunk_id="01-1",
            snippet="intro",
            score=90.0,
            tags=["agent"],
        ),
        RetrievedChunk(
            source="/tmp/frameworks.md",
            title="02 Explore Agentic Frameworks 学习摘要",
            chunk_id="02-1",
            snippet="frameworks",
            score=80.0,
            tags=["agent"],
        ),
        RetrievedChunk(
            source="/tmp/protocols.md",
            title="11 Agentic Protocols 学习摘要",
            chunk_id="11-1",
            snippet="protocols",
            score=70.0,
            tags=["agent"],
        ),
    ]

    ranked = agent.post_rank_study_plan_results(
        "如果我只学 1-* 和 2-*，想做一个 AIAgent 项目，请按课程模块给我安排学习顺序。",
        retrieved_chunks,
    )

    assert [item.title for item in ranked] == [
        "11 Agentic Protocols 学习摘要",
        "02 Explore Agentic Frameworks 学习摘要",
        "01 Intro To AI Agents 学习摘要",
    ]
