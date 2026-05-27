from types import SimpleNamespace

from app.config import validate_settings


def make_settings(
    llm_provider: str = "github",
    api_key: str = "fake-key",
    retrieval_mode: str = "chunk",
    summary_strategy: str = "same-source",
    embedding_provider: str = "hash",
    reranker_provider: str = "none",
    hybrid_candidate_multiplier: int = 3,
    hybrid_candidate_minimum: int = 10,
) -> SimpleNamespace:
    # 用简单对象模拟 Settings，只覆盖校验所需字段
    return SimpleNamespace(
        llm_provider=llm_provider,
        model_name="gpt-4.1-mini",
        api_key=api_key,
        base_url="https://models.inference.ai.azure.com/",
        course_source_root="/tmp",
        retrieval_top_k=5,
        hybrid_candidate_multiplier=hybrid_candidate_multiplier,
        hybrid_candidate_minimum=hybrid_candidate_minimum,
        retrieval_mode=retrieval_mode,
        summary_strategy=summary_strategy,
        embedding_provider=embedding_provider,
        reranker_provider=reranker_provider,
    )


def test_validate_settings_accepts_chunk_retrieval_mode():
    # retrieval_mode=chunk 应通过校验
    settings = make_settings(retrieval_mode="chunk")

    validate_settings(settings)


def test_validate_settings_accepts_document_retrieval_mode():
    # retrieval_mode=document 应通过校验
    settings = make_settings(retrieval_mode="document")

    validate_settings(settings)


def test_validate_settings_accepts_vector_retrieval_mode():
    # retrieval_mode=vector 作为未来预留接口，也应通过配置校验
    settings = make_settings(retrieval_mode="vector")

    validate_settings(settings)


def test_validate_settings_accepts_hybrid_retrieval_mode():
    # retrieval_mode=hybrid 应通过配置校验
    settings = make_settings(retrieval_mode="hybrid")

    validate_settings(settings)


def test_validate_settings_rejects_invalid_retrieval_mode():
    # retrieval_mode 非法值应被明确拒绝
    settings = make_settings(retrieval_mode="invalid-mode")

    try:
        validate_settings(settings)
        assert False, "validate_settings should raise ValueError for invalid retrieval_mode"
    except ValueError as exc:
        assert "RETRIEVAL_MODE" in str(exc)


def test_validate_settings_rejects_invalid_embedding_provider():
    # embedding_provider 非法值应被明确拒绝
    settings = make_settings(embedding_provider="invalid-provider")

    try:
        validate_settings(settings)
        assert False, "validate_settings should raise ValueError for invalid embedding_provider"
    except ValueError as exc:
        assert "EMBEDDING_PROVIDER" in str(exc)

def test_validate_settings_accepts_sentence_transformers_reranker():
    settings = make_settings(reranker_provider="sentence-transformers")
    validate_settings(settings)


def test_validate_settings_rejects_invalid_reranker_provider():
    settings = make_settings(reranker_provider="invalid-reranker")

    try:
        validate_settings(settings)
        assert False, "validate_settings should raise ValueError for invalid reranker_provider"
    except ValueError as exc:
        assert "RERANKER_PROVIDER" in str(exc)


def test_validate_settings_rejects_invalid_hybrid_candidate_multiplier():
    settings = make_settings(hybrid_candidate_multiplier=0)

    try:
        validate_settings(settings)
        assert False, "validate_settings should raise ValueError for invalid hybrid_candidate_multiplier"
    except ValueError as exc:
        assert "HYBRID_CANDIDATE_MULTIPLIER" in str(exc)


def test_validate_settings_rejects_invalid_hybrid_candidate_minimum():
    settings = make_settings(hybrid_candidate_minimum=0)

    try:
        validate_settings(settings)
        assert False, "validate_settings should raise ValueError for invalid hybrid_candidate_minimum"
    except ValueError as exc:
        assert "HYBRID_CANDIDATE_MINIMUM" in str(exc)


def test_validate_settings_rejects_invalid_summary_strategy():
    settings = make_settings(summary_strategy="invalid-summary-strategy")

    try:
        validate_settings(settings)
        assert False, "validate_settings should raise ValueError for invalid summary_strategy"
    except ValueError as exc:
        assert "SUMMARY_STRATEGY" in str(exc)
