from app.schemas import DocumentChunk, RetrievedChunk
from app.vector_store import FakeVectorStore


def make_retrieved_chunk(title: str = "Vector 命中结果") -> RetrievedChunk:
    return RetrievedChunk(
        source="/tmp/vector-source.md",
        title=title,
        chunk_id="vector-chunk-1",
        snippet="向量检索命中的片段",
        score=9.5,
        tags=["agent"],
    )


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
