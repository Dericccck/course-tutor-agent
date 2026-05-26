from types import SimpleNamespace

from app.schemas import DocumentChunk
from app.vector_index_cache import (
    build_chunk_fingerprint,
    build_vector_index_payload,
    is_vector_index_cache_valid,
)


def make_chunk(
    source: str = "/tmp/test.md",
    title: str = "Test Title",
    chunk_id: str = "test-chunk-1",
    content: str = "chunk content",
    tags: list[str] | None = None,
) -> DocumentChunk:
    return DocumentChunk(
        source=source,
        title=title,
        chunk_id=chunk_id,
        content=content,
        tags=tags or ["agent"],
    )


def make_settings(
    embedding_provider: str = "sentence-transformers",
    embedding_model_name: str = "BAAI/bge-m3",
) -> SimpleNamespace:
    return SimpleNamespace(
        embedding_provider=embedding_provider,
        embedding_model_name=embedding_model_name,
        course_source_root="/tmp/course-root",
        course_include_dirs=["2-3-ai-agents-for-beginners"],
    )


def test_build_chunk_fingerprint_is_stable_for_same_chunks():
    chunks = [
        make_chunk(),
        make_chunk(source="/tmp/other.md", chunk_id="other-chunk-1", content="other content"),
    ]

    fingerprint_1 = build_chunk_fingerprint(chunks)
    fingerprint_2 = build_chunk_fingerprint(chunks)

    assert fingerprint_1 == fingerprint_2


def test_build_chunk_fingerprint_changes_when_chunk_content_changes():
    chunks = [make_chunk(content="old content")]
    changed_chunks = [make_chunk(content="new content")]

    fingerprint_1 = build_chunk_fingerprint(chunks)
    fingerprint_2 = build_chunk_fingerprint(changed_chunks)

    assert fingerprint_1 != fingerprint_2


def test_is_vector_index_cache_valid_returns_true_when_payload_matches():
    settings = make_settings()
    chunks = [make_chunk()]
    payload = build_vector_index_payload(
        settings,
        chunks,
        embeddings=[[0.1, 0.2, 0.3]],
    )

    assert is_vector_index_cache_valid(payload, settings, chunks) is True


def test_is_vector_index_cache_valid_returns_false_when_model_changes():
    settings = make_settings()
    chunks = [make_chunk()]
    payload = build_vector_index_payload(
        settings,
        chunks,
        embeddings=[[0.1, 0.2, 0.3]],
    )

    changed_settings = make_settings(embedding_model_name="BAAI/bge-large-zh-v1.5")

    assert is_vector_index_cache_valid(payload, changed_settings, chunks) is False
