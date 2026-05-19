from app.schemas import DocumentChunk, RetrievedChunk
from app.embedding_provider import (
    NotImplementedEmbeddingProvider,
    build_embedding_text,
)
from app.vector_store import FakeVectorStore, InMemoryVectorStore


def make_retrieved_chunk(title: str = "Vector 命中结果") -> RetrievedChunk:
    return RetrievedChunk(
        source="/tmp/vector-source.md",
        title=title,
        chunk_id="vector-chunk-1",
        snippet="向量检索命中的片段",
        score=9.5,
        tags=["agent"],
    )


class MockEmbeddingProvider:
    def __init__(self):
        self.texts: list[str] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.texts = list(texts)
        return [
            [1.0, 0.0],  # 第 1 个 chunk
            [2.0, 0.0],  # 第 2 个 chunk
            [3.0, 0.0],  # 第 3 个 chunk
        ][: len(texts)]

    def embed_query(self, query: str) -> list[float]:
        return [1.0, 0.0]


def test_fake_vector_store_index_chunks_stores_input():
    # FakeVectorStore 应保存传入的 chunks，便于后续本地测试和调试
    store = FakeVectorStore()
    chunks = [
        DocumentChunk(
            source="/tmp/test.md",
            title="Test Title",
            chunk_id="test-chunk-1",
            content="chunk content",
            tags=["agent"],
        )
    ]

    store.index_chunks(chunks)

    assert store.indexed_chunks == chunks


def test_fake_vector_store_search_returns_top_k_results():
    # search 应返回预设结果，并遵守 top_k 截断
    results = [
        make_retrieved_chunk("命中结果 1"),
        make_retrieved_chunk("命中结果 2"),
        make_retrieved_chunk("命中结果 3"),
    ]
    store = FakeVectorStore(results=results)

    found = store.search("tool use", top_k=2)

    assert len(found) == 2
    assert found[0].title == "命中结果 1"
    assert found[1].title == "命中结果 2"


def test_build_embedding_text_includes_title_tags_and_content():
    # embedding 输入文本应同时包含标题、标签和正文
    chunk = DocumentChunk(
        source="/tmp/test.md",
        title="04 Tool Use 学习摘要",
        chunk_id="04-tool-use-chunk-1",
        content="这里是 tool use 的正文内容。",
        tags=["agent", "rag"],
    )

    text = build_embedding_text(chunk)

    assert "标题：04 Tool Use 学习摘要" in text
    assert "标签：agent, rag" in text
    assert "内容：" in text
    assert "这里是 tool use 的正文内容。" in text


def test_build_embedding_text_uses_none_when_tags_empty():
    # tags 为空时，应显式写成“无”，保证格式稳定
    chunk = DocumentChunk(
        source="/tmp/test.md",
        title="无标签测试",
        chunk_id="no-tags-chunk-1",
        content="正文内容",
        tags=[],
    )

    text = build_embedding_text(chunk)

    assert "标题：无标签测试" in text
    assert "标签：无" in text
    assert "正文内容" in text


def test_not_implemented_embedding_provider_embed_texts_raises():
    # 未实现的 embedding provider 应明确提示当前不可用
    provider = NotImplementedEmbeddingProvider()

    try:
        provider.embed_texts(["hello", "world"])
        assert False, "embed_texts should raise NotImplementedError"
    except NotImplementedError as exc:
        assert "Embedding provider is not implemented yet." in str(exc)


def test_not_implemented_embedding_provider_embed_query_raises():
    # 单条查询向量化也应明确提示当前不可用
    provider = NotImplementedEmbeddingProvider()

    try:
        provider.embed_query("tool use")
        assert False, "embed_query should raise NotImplementedError"
    except NotImplementedError as exc:
        assert "Embedding provider is not implemented yet." in str(exc)


def test_in_memory_vector_store_indexes_chunks_with_embedding_provider():
    # 建索引时，应先把 chunk 转成 embedding 文本，再交给 provider 批量向量化
    provider = MockEmbeddingProvider()
    store = InMemoryVectorStore(provider)
    chunks = [
        DocumentChunk(
            source="/tmp/test-1.md",
            title="标题一",
            chunk_id="chunk-1",
            content="正文一",
            tags=["agent"],
        ),
        DocumentChunk(
            source="/tmp/test-2.md",
            title="标题二",
            chunk_id="chunk-2",
            content="正文二",
            tags=["rag"],
        ),
    ]

    store.index_chunks(chunks)

    assert store.chunks == chunks
    assert len(store.embeddings) == 2
    assert "标题：标题一" in provider.texts[0]
    assert "正文二" in provider.texts[1]


def test_in_memory_vector_store_search_returns_ranked_results():
    # search 应根据 query embedding 和 chunk embeddings 的分数返回排序后的结果
    provider = MockEmbeddingProvider()
    store = InMemoryVectorStore(provider)

    chunks = [
        DocumentChunk(
            source="/tmp/a.md",
            title="A",
            chunk_id="a-1",
            content="内容 A",
            tags=["agent"],
        ),
        DocumentChunk(
            source="/tmp/b.md",
            title="B",
            chunk_id="b-1",
            content="内容 B",
            tags=["rag"],
        ),
        DocumentChunk(
            source="/tmp/c.md",
            title="C",
            chunk_id="c-1",
            content="内容 C",
            tags=["agent"],
        ),
    ]

    store.index_chunks(chunks)

    results = store.search("tool use", top_k=2)

    assert len(results) == 2
    assert results[0].title == "C"
    assert results[1].title == "B"
    assert results[0].chunk_id == "c-1"
    assert results[1].chunk_id == "b-1"
