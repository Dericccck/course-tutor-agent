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
    summary_strategy: str = "same-source",
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
        course_include_dirs=["2-3-ai-agents-for-beginners"],
        retrieval=SimpleNamespace(
            retrieval_top_k=5,
            summary_strategy=summary_strategy,
            hybrid_candidate_multiplier=hybrid_candidate_multiplier,
            hybrid_candidate_minimum=hybrid_candidate_minimum,
            retrieval_mode=retrieval_mode,
            embedding_provider="hash",
            embedding_model_name="BAAI/bge-m3",
            embedding_cache_dir=None,
            reranker_provider="none",
            reranker_model_name="BAAI/bge-reranker-base",
            reranker_cache_dir=None,
        ),
    )


def test_ask_course_agent_returns_fallback_when_no_retrieved_chunks(monkeypatch):
    # 当检索结果为空时，应直接返回兜底回答，而不是继续调模型
    monkeypatch.setattr(agent, "validate_settings", lambda settings: None)
    monkeypatch.setattr(agent, "retrieve_documents", lambda query, documents, top_k: [])
    monkeypatch.setattr(
        agent,
        "load_agent_runtime_config",
        lambda: {
            "fallback": {
                "no_results_answer": "当前没有检索到相关课程资料，暂时无法回答这个问题",
                "no_results_suggestions": ["换一个更具体的问题试试", "优先使用课程名称、章节名或关键词提问"],
            }
        },
    )

    result = agent.ask_course_agent(
        question="一个完全找不到资料的问题",
        documents=[make_document()],
        settings=make_settings(),
        memory={},
    )

    assert "当前没有检索到相关课程资料" in result.answer
    assert result.sources == []


def test_is_rag_study_plan_question_uses_configured_keywords(monkeypatch):
    monkeypatch.setattr(
        agent,
        "load_study_plan_order_config",
        lambda: {
            "route_triggers": {
                "rag_route_keywords": ["agentic rag", "检索增强生成"],
            }
        },
    )

    assert agent.is_rag_study_plan_question("我想重点学 Agentic RAG") is True
    assert agent.is_rag_study_plan_question("我想重点学检索增强生成") is True
    assert agent.is_rag_study_plan_question("我想学多 agent 协作") is False


def test_detect_task_type_uses_configured_keywords(monkeypatch):
    monkeypatch.setattr(
        agent,
        "load_agent_runtime_config",
        lambda: {
            "task_routing": {
                "summary_keywords": ["复盘"],
                "study_plan_keywords": ["路线图"],
            }
        },
    )

    assert agent.detect_task_type("帮我复盘这一节") == "summary"
    assert agent.detect_task_type("请给我一个学习路线图") == "study_plan"
    assert agent.detect_task_type("tool use 是什么") == "qa"


def test_build_retrieval_queries_adds_rule_based_qa_expansion():
    queries = agent.build_retrieval_queries(
        "tool use 是什么？",
        task_type="qa",
        memory={},
    )

    assert queries == [
        "tool use 是什么？",
        "tool use agent tool calling",
        "04 Tool Use 学习摘要",
    ]


def test_build_retrieval_queries_includes_recent_focus_for_qa():
    queries = agent.build_retrieval_queries(
        "tool use 是什么？",
        task_type="qa",
        memory={"recent_focus": "Tool use 与 agent tool calling"},
    )

    assert queries == [
        "tool use 是什么？",
        "tool use agent tool calling",
        "04 Tool Use 学习摘要",
        "Tool use 与 agent tool calling",
    ]


def test_build_retrieval_queries_adds_study_plan_goal_expansion():
    queries = agent.build_retrieval_queries(
        "如果我想重点学 RAG，再过渡到 Agentic RAG，应该怎么安排学习顺序？",
        task_type="study_plan",
        memory={"learning_goal": "我想重点学习 RAG，并进一步过渡到 Agentic RAG"},
    )

    assert queries == [
        "如果我想重点学 RAG，再过渡到 Agentic RAG，应该怎么安排学习顺序？",
        "我想重点学习 RAG，并进一步过渡到 Agentic RAG 学习顺序 学习路线",
        "05 Agentic RAG 学习摘要",
    ]


def test_build_retrieval_queries_includes_recent_focus_for_summary():
    queries = agent.build_retrieval_queries(
        "帮我总结 05-agentic-rag 这一节在讲什么",
        task_type="summary",
        memory={"recent_focus": "05 Agentic RAG 总结"},
    )

    assert queries == [
        "帮我总结 05-agentic-rag 这一节在讲什么",
        "帮我总结 05-agentic-rag 这一节在讲什么 notebook lesson 总结",
        "05 Agentic RAG 学习摘要",
        "05 Agentic RAG 总结 lesson notebook 总结",
    ]


def test_build_retrieval_queries_includes_recent_focus_for_study_plan():
    queries = agent.build_retrieval_queries(
        "如果我想重点学 RAG，再过渡到 Agentic RAG，应该怎么安排学习顺序？",
        task_type="study_plan",
        memory={
            "learning_goal": "我想重点学习 RAG，并进一步过渡到 Agentic RAG",
            "recent_focus": "RAG 到 Agentic RAG 学习路线",
        },
    )

    assert queries == [
        "如果我想重点学 RAG，再过渡到 Agentic RAG，应该怎么安排学习顺序？",
        "我想重点学习 RAG，并进一步过渡到 Agentic RAG 学习顺序 学习路线",
        "05 Agentic RAG 学习摘要",
        "RAG 到 Agentic RAG 学习路线 学习顺序 学习路线",
    ]


def test_detect_course_anchor_maps_known_course_titles():
    assert agent.detect_course_anchor("tool use 是什么？") == "04 Tool Use 学习摘要"
    assert agent.detect_course_anchor("帮我总结 05-agentic-rag 这一节在讲什么") == "05 Agentic RAG 学习摘要"
    assert agent.detect_course_anchor("planning agent 是什么？") == "07 Planning Design 学习摘要"
    assert agent.detect_course_anchor("Lesson 1 在讲什么") == "Lesson 1 学习摘要"
    assert agent.detect_course_anchor("一个完全无关的问题") is None


def test_merge_multi_query_results_deduplicates_and_keeps_highest_score():
    result_groups = [
        [
            RetrievedChunk(
                source="/tmp/tool.md",
                title="04 Tool Use 学习摘要",
                chunk_id="tool-1",
                snippet="tool low",
                score=8.0,
                tags=["agent"],
            ),
            RetrievedChunk(
                source="/tmp/framework.md",
                title="02 Explore Agentic Frameworks 学习摘要",
                chunk_id="framework-1",
                snippet="framework",
                score=7.5,
                tags=["agent"],
            ),
        ],
        [
            RetrievedChunk(
                source="/tmp/tool.md",
                title="04 Tool Use 学习摘要",
                chunk_id="tool-1",
                snippet="tool high",
                score=9.5,
                tags=["agent"],
            ),
        ],
    ]

    merged = agent.merge_multi_query_results(result_groups, top_k=5)

    assert [(item.source, item.chunk_id, item.score) for item in merged] == [
        ("/tmp/tool.md", "tool-1", 9.5),
        ("/tmp/framework.md", "framework-1", 7.5),
    ]


def test_should_retry_retrieval_uses_task_specific_thresholds():
    one_chunk = [make_chunk("04 Tool Use 学习摘要")]
    two_chunks = [make_chunk("04 Tool Use 学习摘要"), make_chunk("05 Agentic RAG 学习摘要")]

    assert agent.should_retry_retrieval([], "qa") is True
    assert agent.should_retry_retrieval(one_chunk, "qa") is True
    assert agent.should_retry_retrieval(two_chunks, "qa") is False
    assert agent.should_retry_retrieval(one_chunk, "summary") is True
    assert agent.should_retry_retrieval(two_chunks, "summary") is False
    assert agent.should_retry_retrieval(two_chunks, "study_plan") is True


def test_build_retry_retrieval_queries_includes_goal_scope_and_recent_focus():
    queries = agent.build_retry_retrieval_queries(
        "如果我想重点学 RAG，再过渡到 Agentic RAG，应该怎么安排学习顺序？",
        task_type="study_plan",
        memory={
            "learning_goal": "我想重点学习 RAG，并进一步过渡到 Agentic RAG",
            "preferred_scope": "我只学习 1-* 和 2-* 的内容",
            "recent_focus": "RAG 到 Agentic RAG 学习路线",
        },
    )

    assert queries == [
        "如果我想重点学 RAG，再过渡到 Agentic RAG，应该怎么安排学习顺序？ 学习顺序 roadmap lesson",
        "我想重点学习 RAG，并进一步过渡到 Agentic RAG",
        "我只学习 1-* 和 2-* 的内容",
        "RAG 到 Agentic RAG 学习路线",
    ]


def test_merge_retrieval_results_uses_configured_max_per_source(monkeypatch):
    monkeypatch.setattr(
        agent,
        "load_agent_runtime_config",
        lambda: {
            "retrieval_merge": {
                "max_per_source": 1,
                "prioritize_same_group_secondary": True,
            }
        },
    )

    primary = [
        RetrievedChunk(
            source="/tmp/a.md",
            title="A1",
            chunk_id="a-1",
            snippet="A1",
            score=10.0,
            tags=["agent"],
        ),
        RetrievedChunk(
            source="/tmp/a.md",
            title="A2",
            chunk_id="a-2",
            snippet="A2",
            score=9.5,
            tags=["agent"],
        ),
    ]
    secondary = [
        RetrievedChunk(
            source="/tmp/b.md",
            title="B1",
            chunk_id="b-1",
            snippet="B1",
            score=9.0,
            tags=["agent"],
        )
    ]

    merged = agent.merge_retrieval_results(primary, secondary, top_k=3)

    assert [item.chunk_id for item in merged] == ["a-1", "b-1"]


def test_ask_course_agent_uses_configured_no_results_fallback(monkeypatch):
    monkeypatch.setattr(agent, "validate_settings", lambda settings: None)
    monkeypatch.setattr(agent, "retrieve_documents", lambda query, documents, top_k: [])
    monkeypatch.setattr(
        agent,
        "load_agent_runtime_config",
        lambda: {
            "fallback": {
                "no_results_answer": "自定义兜底回答",
                "no_results_suggestions": ["建议一", "建议二"],
            }
        },
    )

    result = agent.ask_course_agent(
        question="一个完全找不到资料的问题",
        documents=[make_document()],
        settings=make_settings(),
        memory={},
    )

    assert result.answer == "自定义兜底回答"
    assert result.suggestions == ["建议一", "建议二"]


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


def test_ask_course_agent_runs_chunk_retrieval_for_each_query(monkeypatch):
    monkeypatch.setattr(agent, "validate_settings", lambda settings: None)
    seen_queries: list[str] = []

    def fake_retrieve_chunks(query, chunks, top_k):
        seen_queries.append(query)
        return [
            RetrievedChunk(
                source=f"/tmp/{query}.md",
                title=f"{query} 标题",
                chunk_id=f"{query}-chunk-1",
                snippet=query,
                score=10.0 if len(seen_queries) == 1 else 9.0,
                tags=["agent"],
            )
        ]

    monkeypatch.setattr(agent, "retrieve_chunks", fake_retrieve_chunks)
    monkeypatch.setattr(agent, "retrieve_documents", lambda query, documents, top_k: [])

    fake_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content='{"answer": "chunk 多 query 检索结果", "suggestions": [], "sources": []}'
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

    assert seen_queries == [
        "tool use 是什么？",
        "tool use agent tool calling",
        "04 Tool Use 学习摘要",
    ]
    assert result.answer == "chunk 多 query 检索结果"


def test_ask_course_agent_triggers_retry_round_when_first_retrieval_is_weak(monkeypatch):
    monkeypatch.setattr(agent, "validate_settings", lambda settings: None)
    seen_queries: list[str] = []

    def fake_retrieve_chunks(query, chunks, top_k):
        seen_queries.append(query)
        if len(seen_queries) <= 3:
            return [
                RetrievedChunk(
                    source="/tmp/shared-tool.md",
                    title="shared tool 标题",
                    chunk_id="shared-tool-chunk-1",
                    snippet=query,
                    score=10.0,
                    tags=["agent"],
                )
            ]
        return [
            RetrievedChunk(
                source=f"/tmp/{query}.md",
                title=f"{query} 标题",
                chunk_id=f"{query}-chunk-1",
                snippet=query,
                score=9.0,
                tags=["agent"],
            ),
            RetrievedChunk(
                source=f"/tmp/{query}-extra.md",
                title=f"{query} 扩展标题",
                chunk_id=f"{query}-chunk-2",
                snippet=query,
                score=8.5,
                tags=["agent"],
            ),
        ]

    monkeypatch.setattr(agent, "retrieve_chunks", fake_retrieve_chunks)
    monkeypatch.setattr(agent, "retrieve_documents", lambda query, documents, top_k: [])

    fake_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content='{"answer": "触发 retry 的结果", "suggestions": [], "sources": []}'
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

    assert seen_queries == [
        "tool use 是什么？",
        "tool use agent tool calling",
        "04 Tool Use 学习摘要",
        "tool use 是什么？ agent course concept",
    ]
    assert result.answer == "触发 retry 的结果"
    assert result.debug["task_type"] == "qa"
    assert result.debug["retry_triggered"] is True
    assert result.debug["initial_queries"] == [
        "tool use 是什么？",
        "tool use agent tool calling",
        "04 Tool Use 学习摘要",
    ]
    assert result.debug["retry_queries"] == [
        "tool use 是什么？ agent course concept",
    ]
    assert result.debug["initial_result_count"] == 1
    assert result.debug["final_result_count"] == 3


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
        assert "vector mode requires vector_store" in str(exc)


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


def test_ask_course_agent_keeps_only_model_cited_allowed_sources(monkeypatch):
    monkeypatch.setattr(agent, "validate_settings", lambda settings: None)
    monkeypatch.setattr(
        agent,
        "retrieve_documents",
        lambda query, documents, top_k: [
            make_chunk("04 Tool Use 学习摘要"),
            make_chunk("05 Agentic RAG 学习摘要"),
        ],
    )

    fake_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=json.dumps(
                        {
                            "answer": "回答里主要使用了前两条资料。",
                            "suggestions": [],
                            "sources": [
                                "/tmp/04 Tool Use 学习摘要.md",
                                "/tmp/不会被允许的来源.md",
                            ],
                        }
                    )
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
    )

    assert result.sources == ["/tmp/04 Tool Use 学习摘要.md"]


def test_ask_course_agent_falls_back_to_prompt_sources_when_model_sources_invalid(monkeypatch):
    monkeypatch.setattr(agent, "validate_settings", lambda settings: None)
    monkeypatch.setattr(
        agent,
        "retrieve_documents",
        lambda query, documents, top_k: [
            make_chunk("04 Tool Use 学习摘要"),
            make_chunk("05 Agentic RAG 学习摘要"),
        ],
    )

    fake_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=json.dumps(
                        {
                            "answer": "回答内容",
                            "suggestions": [],
                            "sources": ["/tmp/不存在的来源.md"],
                        }
                    )
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
    )

    assert result.sources == [
        "/tmp/04 Tool Use 学习摘要.md",
        "/tmp/05 Agentic RAG 学习摘要.md",
    ]


def test_select_prompt_chunks_limits_qa_context_size():
    chunks = [make_chunk(f"标题 {index}") for index in range(1, 6)]

    selected = agent.select_prompt_chunks("qa", chunks)

    assert len(selected) == 3
    assert [item.title for item in selected] == ["标题 1", "标题 2", "标题 3"]


def test_normalize_suggestions_deduplicates_filters_empty_and_limits_to_three():
    result = agent.normalize_suggestions(
        [
            "深入学习 tool use 的调用方式",
            "",
            "深入学习 tool use 的调用方式",
            "阅读 04-tool-use 相关 notebook",
            "结合规划设计理解工具调用",
            "额外的第四条建议",
        ],
        question="tool use 是什么？",
    )

    assert result == [
        "深入学习 tool use 的调用方式",
        "阅读 04-tool-use 相关 notebook",
        "结合规划设计理解工具调用",
    ]


def test_normalize_suggestions_filters_question_text():
    result = agent.normalize_suggestions(
        [
            "tool use 是什么？",
            "阅读 04-tool-use 相关 notebook",
        ],
        question="tool use 是什么？",
    )

    assert result == ["阅读 04-tool-use 相关 notebook"]


def test_normalize_answer_text_strips_whitespace():
    result = agent.normalize_answer_text("  正常回答内容  ")

    assert result == "正常回答内容"


def test_normalize_answer_text_returns_fallback_for_empty_text():
    result = agent.normalize_answer_text("   ")

    assert result == "根据当前资料，暂时无法生成稳定答案。"


def test_ask_course_agent_normalizes_model_suggestions(monkeypatch):
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
                    content=json.dumps(
                        {
                            "answer": "正常回答",
                            "suggestions": [
                                "tool use 是什么？",
                                "深入学习 tool use 的调用方式",
                                "",
                                "深入学习 tool use 的调用方式",
                                "阅读 04-tool-use 相关 notebook",
                                "结合规划设计理解工具调用",
                            ],
                            "sources": ["/tmp/04 Tool Use 学习摘要.md"],
                        }
                    )
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
    )

    assert result.suggestions == [
        "深入学习 tool use 的调用方式",
        "阅读 04-tool-use 相关 notebook",
        "结合规划设计理解工具调用",
    ]


def test_ask_course_agent_normalizes_empty_model_answer(monkeypatch):
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
                    content=json.dumps(
                        {
                            "answer": "   ",
                            "suggestions": [],
                            "sources": ["/tmp/04 Tool Use 学习摘要.md"],
                        }
                    )
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
    )

    assert result.answer == "根据当前资料，暂时无法生成稳定答案。"


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
        assert "hybrid mode requires vector_store" in str(exc)


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

    narrowed = agent.narrow_summary_results(
        retrieved_chunks,
        strategy="same-source",
    )

    assert len(narrowed) == 2
    assert all(item.source == "/tmp/07-planning-design.md" for item in narrowed)


def test_narrow_summary_results_returns_original_results_for_unknown_strategy():
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
            source="/tmp/other.md",
            title="Other",
            chunk_id="other-1",
            snippet="other",
            score=8.0,
            tags=["rag"],
        ),
    ]

    narrowed = agent.narrow_summary_results(
        retrieved_chunks,
        strategy="unknown-strategy",
    )

    assert narrowed == retrieved_chunks


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


def test_ask_course_agent_merges_multi_query_hybrid_results_before_rerank(monkeypatch):
    monkeypatch.setattr(agent, "validate_settings", lambda settings: None)
    lexical_queries: list[str] = []
    vector_queries: list[str] = []

    def fake_retrieve_chunks(query, chunks, top_k):
        lexical_queries.append(query)
        return [
            RetrievedChunk(
                source=f"/tmp/lexical-{len(lexical_queries)}.md",
                title=f"lexical-{len(lexical_queries)}",
                chunk_id=f"lexical-{len(lexical_queries)}",
                snippet=query,
                score=10.0 - len(lexical_queries),
                tags=["agent"],
            )
        ]

    class FakeVectorStore:
        def search(self, query, top_k=5):
            vector_queries.append(query)
            return [
                RetrievedChunk(
                    source=f"/tmp/vector-{len(vector_queries)}.md",
                    title=f"vector-{len(vector_queries)}",
                    chunk_id=f"vector-{len(vector_queries)}",
                    snippet=query,
                    score=8.0 - len(vector_queries),
                    tags=["agent"],
                )
            ]

    monkeypatch.setattr(agent, "retrieve_chunks", fake_retrieve_chunks)

    fake_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content='{"answer": "hybrid 多 query 结果", "suggestions": [], "sources": []}'
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
                source="/tmp/reranked.md",
                title="reranked",
                chunk_id="reranked-1",
                snippet="reranked result",
                score=99.0,
                tags=["agent"],
            )
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

    assert lexical_queries == [
        "tool use 是什么？",
        "tool use agent tool calling",
        "04 Tool Use 学习摘要",
        "tool use 是什么？ agent course concept",
    ]
    assert vector_queries == [
        "tool use 是什么？",
        "tool use agent tool calling",
        "04 Tool Use 学习摘要",
        "tool use 是什么？ agent course concept",
    ]
    assert len(reranker.calls) == 2
    assert len(reranker.calls[0][1]) == 5
    assert len(reranker.calls[1][1]) == 2
    assert result.answer == "hybrid 多 query 结果"


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

    assert lexical_top_ks == [20, 20, 20]
    assert vector_top_ks == [20, 20, 20]
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
