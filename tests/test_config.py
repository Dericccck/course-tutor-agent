from types import SimpleNamespace

from app.config import validate_settings


def make_settings(
    llm_provider: str = "github",
    api_key: str = "fake-key",
    retrieval_mode: str = "chunk",
) -> SimpleNamespace:
    # 用简单对象模拟 Settings，只覆盖校验所需字段
    return SimpleNamespace(
        llm_provider=llm_provider,
        model_name="gpt-4.1-mini",
        api_key=api_key,
        base_url="https://models.inference.ai.azure.com/",
        course_source_root="/tmp",
        retrieval_top_k=5,
        retrieval_mode=retrieval_mode,
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


def test_validate_settings_rejects_invalid_retrieval_mode():
    # retrieval_mode 非法值应被明确拒绝
    settings = make_settings(retrieval_mode="invalid-mode")

    try:
        validate_settings(settings)
        assert False, "validate_settings should raise ValueError for invalid retrieval_mode"
    except ValueError as exc:
        assert "RETRIEVAL_MODE" in str(exc)
