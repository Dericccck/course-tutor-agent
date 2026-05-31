import importlib.util
from pathlib import Path


RUN_EVAL_PATH = (
    Path(__file__).resolve().parents[1] / "eval" / "run_eval.py"
)


def load_run_eval_module():
    spec = importlib.util.spec_from_file_location(
        "eval_run_eval",
        RUN_EVAL_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_should_run_agent_eval_defaults_to_false(monkeypatch):
    monkeypatch.delenv("RUN_AGENT_EVAL", raising=False)
    run_eval = load_run_eval_module()

    assert run_eval.should_run_agent_eval() is False


def test_should_run_agent_eval_returns_false_for_disabled_values(monkeypatch):
    run_eval = load_run_eval_module()

    for value in ["false", "0", "off", "no"]:
        monkeypatch.setenv("RUN_AGENT_EVAL", value)
        assert run_eval.should_run_agent_eval() is False


def test_should_run_agent_eval_returns_true_for_enabled_values(monkeypatch):
    run_eval = load_run_eval_module()

    for value in ["true", "1", "yes", "on"]:
        monkeypatch.setenv("RUN_AGENT_EVAL", value)
        assert run_eval.should_run_agent_eval() is True


def test_get_eval_modes_defaults_to_chunk_vector_hybrid(monkeypatch):
    monkeypatch.delenv("EVAL_MODES", raising=False)
    run_eval = load_run_eval_module()

    assert run_eval.get_eval_modes() == ["chunk", "vector", "hybrid"]


def test_get_eval_modes_supports_subset(monkeypatch):
    monkeypatch.setenv("EVAL_MODES", "chunk,hybrid")
    run_eval = load_run_eval_module()

    assert run_eval.get_eval_modes() == ["chunk", "hybrid"]


def test_get_eval_modes_rejects_invalid_values(monkeypatch):
    monkeypatch.setenv("EVAL_MODES", "chunk,invalid-mode")
    run_eval = load_run_eval_module()

    try:
        run_eval.get_eval_modes()
        assert False, "get_eval_modes should raise ValueError for invalid mode"
    except ValueError as exc:
        assert "unsupported modes" in str(exc)


def test_get_eval_tags_defaults_to_none(monkeypatch):
    monkeypatch.delenv("EVAL_TAGS", raising=False)
    run_eval = load_run_eval_module()

    assert run_eval.get_eval_tags() is None


def test_get_eval_tags_returns_normalized_tag_set(monkeypatch):
    monkeypatch.setenv("EVAL_TAGS", "study_plan, RAG ")
    run_eval = load_run_eval_module()

    assert run_eval.get_eval_tags() == {"study_plan", "rag"}


def test_should_run_agent_eval_uses_config_default(monkeypatch):
    monkeypatch.delenv("RUN_AGENT_EVAL", raising=False)
    run_eval = load_run_eval_module()
    monkeypatch.setattr(
        run_eval,
        "load_eval_config",
        lambda: {"default_run_agent_eval": True},
    )

    assert run_eval.should_run_agent_eval() is True


def test_get_eval_tags_uses_config_default(monkeypatch):
    monkeypatch.delenv("EVAL_TAGS", raising=False)
    run_eval = load_run_eval_module()
    monkeypatch.setattr(
        run_eval,
        "load_eval_config",
        lambda: {"default_tags": ["study_plan", "rag"]},
    )

    assert run_eval.get_eval_tags() == {"study_plan", "rag"}


def test_is_mode_enabled_for_sample_returns_true_when_field_missing():
    run_eval = load_run_eval_module()
    sample = {"id": "sample-without-enabled-modes"}

    assert run_eval.is_mode_enabled_for_sample(sample, "chunk") is True
    assert run_eval.is_mode_enabled_for_sample(sample, "hybrid") is True


def test_is_mode_enabled_for_sample_respects_enabled_modes():
    run_eval = load_run_eval_module()
    sample = {
        "id": "study-plan-rag-001",
        "enabled_modes": ["vector", "hybrid"],
    }

    assert run_eval.is_mode_enabled_for_sample(sample, "vector") is True
    assert run_eval.is_mode_enabled_for_sample(sample, "hybrid") is True
    assert run_eval.is_mode_enabled_for_sample(sample, "chunk") is False


def test_is_tag_enabled_for_sample_returns_true_when_filter_missing():
    run_eval = load_run_eval_module()
    sample = {"id": "sample-without-tag-filter", "tags": ["qa", "agent-course"]}

    assert run_eval.is_tag_enabled_for_sample(sample, None) is True


def test_is_tag_enabled_for_sample_matches_on_intersection():
    run_eval = load_run_eval_module()
    sample = {"id": "study-plan-rag-001", "tags": ["study_plan", "rag", "cross-course"]}

    assert run_eval.is_tag_enabled_for_sample(sample, {"rag"}) is True
    assert run_eval.is_tag_enabled_for_sample(sample, {"study_plan", "summary"}) is True
    assert run_eval.is_tag_enabled_for_sample(sample, {"qa"}) is False


def test_evalvate_answer_sources_matches_expected_source_fragments():
    run_eval = load_run_eval_module()
    sample = {
        "expected_sources_contains": [
            "2-3-ai-agents-for-beginners/04-tool-use/notebook-summary.md",
            "2-3-ai-agents-for-beginners/05-agentic-rag/notebook-summary.md",
        ]
    }
    answer_sources = [
        "/Users/a1-6/Desktop/AIAgent/code/2-3-ai-agents-for-beginners/04-tool-use/notebook-summary.md#chunk-1",
        "/Users/a1-6/Desktop/AIAgent/code/2-3-ai-agents-for-beginners/05-agentic-rag/notebook-summary.md#chunk-2",
    ]

    result = run_eval.evalvate_answer_sources(sample, answer_sources)

    assert result["source_citation_hit"] is True
    assert result["source_hits"] == sample["expected_sources_contains"]


def test_evalvate_answer_sources_returns_none_when_not_configured():
    run_eval = load_run_eval_module()

    result = run_eval.evalvate_answer_sources(
        {"id": "sample-without-source-expectation"},
        ["/tmp/example.md#chunk-1"],
    )

    assert result["expected_sources_contains"] == []
    assert result["source_hits"] == []
    assert result["source_citation_hit"] is None


def test_should_run_agent_sample_respects_selected_sample_whitelist():
    run_eval = load_run_eval_module()

    assert run_eval.should_run_agent_sample({"id": "study-plan-rag-001"}) is True
    assert run_eval.should_run_agent_sample({"id": "qa-tool-use-001"}) is True
    assert run_eval.should_run_agent_sample({"id": "summary-agentic-rag-001"}) is True
    assert run_eval.should_run_agent_sample({"id": "qa-planning-agent-001"}) is False
