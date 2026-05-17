from app.loader import load_document_chunks
from app.retriever import retrieve_chunks


ROOT_DIR = "/Users/a1-6/Desktop/AIAgent/code"


def test_load_document_chunks_returns_more_than_documents():
    # chunk 数量应该明显大于原始文档数量，说明切块已经生效
    chunks = load_document_chunks(ROOT_DIR)

    assert len(chunks) > 48


def test_retrieve_chunks_for_summary_query_hits_target_section():
    # 总结章节时，应优先命中对应章节的 chunk
    chunks = load_document_chunks(ROOT_DIR)
    query = "帮我总结 07-planning-design 这一节在讲什么"

    results = retrieve_chunks(query, chunks, top_k=5)
    sources = [item.source for item in results]

    assert any("07-planning-design" in source for source in sources)


def test_retrieve_chunks_for_tool_use_query_hits_expected_title():
    # tool use 问题应优先命中对应模块
    chunks = load_document_chunks(ROOT_DIR)
    query = "tool use 是什么？"

    results = retrieve_chunks(query, chunks, top_k=5)
    titles = [item.title for item in results]

    assert any("04 Tool Use" in title for title in titles)


def test_retrieve_chunks_limits_same_source_results():
    # 同一 source 最多保留 2 个 chunk，避免一个文档占满结果位
    chunks = load_document_chunks(ROOT_DIR)
    query = "tool use 是什么？"

    results = retrieve_chunks(query, chunks, top_k=5)

    source_counts: dict[str, int] = {}
    for item in results:
        source_counts[item.source] = source_counts.get(item.source, 0) + 1

    assert all(count <= 2 for count in source_counts.values())


def test_retrieve_chunks_returns_non_empty_snippets():
    # 返回的 snippet 应该非空，确保 chunk 检索结果能直接用于后续 prompt
    chunks = load_document_chunks(ROOT_DIR)
    query = "tool use 是什么？"

    results = retrieve_chunks(query, chunks, top_k=5)

    assert results
    assert all(item.snippet.strip() for item in results)


def test_retrieve_chunks_sets_real_chunk_ids():
    # chunk 检索路径应返回真实 chunk_id，方便后续做更细粒度引用
    chunks = load_document_chunks(ROOT_DIR)
    query = "tool use 是什么？"

    results = retrieve_chunks(query, chunks, top_k=5)

    assert results
    assert all(item.chunk_id for item in results)
