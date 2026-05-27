from types import SimpleNamespace

from app.schemas import DocumentChunk
from app import vector_index_service


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
    embedding_cache_dir: str | None = "/tmp/models",
    vector_index_cache_path: str | None = "/tmp/vector-index-cache.json",
) -> SimpleNamespace:
    return SimpleNamespace(
        embedding_provider=embedding_provider,
        embedding_model_name=embedding_model_name,
        embedding_cache_dir=embedding_cache_dir,
        vector_index_cache_path=vector_index_cache_path,
    )


class FakeVectorStore:
    def __init__(self, embedding_provider):
        self.embedding_provider = embedding_provider
        self.load_index_calls: list[tuple[list[DocumentChunk], list[list[float]]]] = []
        self.index_chunks_calls: list[list[DocumentChunk]] = []
        self.embeddings: list[list[float]] = []

    def load_index(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> None:
        self.load_index_calls.append((list(chunks), list(embeddings)))
        self.chunks = list(chunks)
        self.embeddings = list(embeddings)

    def index_chunks(self, chunks: list[DocumentChunk]) -> None:
        self.index_chunks_calls.append(list(chunks))
        self.chunks = list(chunks)
        self.embeddings = [[0.1, 0.2] for _ in chunks]


def test_build_vector_store_with_cache_loads_existing_index(monkeypatch):
    settings = make_settings()
    chunks = [make_chunk()]
    embedding_provider = object()
    payload = {
        "chunks": [
            {
                "source": chunks[0].source,
                "title": chunks[0].title,
                "chunk_id": chunks[0].chunk_id,
                "content": chunks[0].content,
                "tags": chunks[0].tags,
            }
        ],
        "embeddings": [[0.9, 0.1]],
    }

    monkeypatch.setattr(
        vector_index_service,
        "build_embedding_provider",
        lambda *args, **kwargs: embedding_provider,
    )
    monkeypatch.setattr(vector_index_service, "InMemoryVectorStore", FakeVectorStore)
    monkeypatch.setattr(
        vector_index_service,
        "load_vector_index_cache",
        lambda cache_path: payload,
    )
    monkeypatch.setattr(
        vector_index_service,
        "is_vector_index_cache_valid",
        lambda cache_payload, active_settings, active_chunks: True,
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("should not rebuild or save when cache is valid")

    monkeypatch.setattr(vector_index_service, "build_vector_index_payload", fail_if_called)
    monkeypatch.setattr(vector_index_service, "save_vector_index_cache", fail_if_called)

    store = vector_index_service.build_vector_store_with_cache(settings, chunks)

    assert store.embedding_provider is embedding_provider
    assert len(store.load_index_calls) == 1
    assert store.index_chunks_calls == []
    restored_chunks, restored_embeddings = store.load_index_calls[0]
    assert len(restored_chunks) == 1
    assert restored_chunks[0].source == chunks[0].source
    assert restored_chunks[0].title == chunks[0].title
    assert restored_chunks[0].chunk_id == chunks[0].chunk_id
    assert restored_chunks[0].content == chunks[0].content
    assert restored_chunks[0].tags == chunks[0].tags
    assert restored_embeddings == [[0.9, 0.1]]


def test_build_vector_store_with_cache_rebuilds_and_saves_when_cache_missing(monkeypatch):
    settings = make_settings()
    chunks = [make_chunk()]
    embedding_provider = object()
    saved_payloads: list[tuple[str, dict]] = []

    monkeypatch.setattr(
        vector_index_service,
        "build_embedding_provider",
        lambda *args, **kwargs: embedding_provider,
    )
    monkeypatch.setattr(vector_index_service, "InMemoryVectorStore", FakeVectorStore)
    monkeypatch.setattr(
        vector_index_service,
        "load_vector_index_cache",
        lambda cache_path: None,
    )
    monkeypatch.setattr(
        vector_index_service,
        "is_vector_index_cache_valid",
        lambda cache_payload, active_settings, active_chunks: False,
    )
    monkeypatch.setattr(
        vector_index_service,
        "build_vector_index_payload",
        lambda active_settings, active_chunks, embeddings: {
            "chunks": active_chunks,
            "embeddings": embeddings,
        },
    )
    monkeypatch.setattr(
        vector_index_service,
        "save_vector_index_cache",
        lambda cache_path, payload: saved_payloads.append((cache_path, payload)),
    )

    store = vector_index_service.build_vector_store_with_cache(settings, chunks)

    assert len(store.load_index_calls) == 0
    assert len(store.index_chunks_calls) == 1
    assert store.index_chunks_calls[0] == chunks
    assert saved_payloads == [
        (
            settings.vector_index_cache_path,
            {
                "chunks": chunks,
                "embeddings": [[0.1, 0.2]],
            },
        )
    ]
