import importlib.util
import json
from collections import Counter
from pathlib import Path


ANALYZE_TRACES_PATH = (
    Path(__file__).resolve().parents[1] / "eval" / "analyze_traces.py"
)


def load_analyze_traces_module():
    spec = importlib.util.spec_from_file_location(
        "eval_analyze_traces",
        ANALYZE_TRACES_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_load_jsonl_returns_empty_list_when_file_missing(tmp_path: Path):
    analyze_traces = load_analyze_traces_module()

    result = analyze_traces.load_jsonl(tmp_path / "missing.jsonl")

    assert result == []


def test_load_jsonl_reads_non_empty_lines(tmp_path: Path):
    analyze_traces = load_analyze_traces_module()
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        json.dumps({"id": 1}, ensure_ascii=False) + "\n\n" + json.dumps({"id": 2}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    result = analyze_traces.load_jsonl(trace_path)

    assert result == [{"id": 1}, {"id": 2}]


def test_analyze_cli_traces_summarizes_retry_and_task_distribution():
    analyze_traces = load_analyze_traces_module()
    records = [
        {
            "debug": {
                "task_type": "qa",
                "retry_triggered": False,
                "llm_retry_query": None,
            }
        },
        {
            "debug": {
                "task_type": "summary",
                "retry_triggered": True,
                "llm_retry_query": "summary notebook lesson summary",
            }
        },
    ]

    summary = analyze_traces.analyze_cli_traces(records)

    assert summary["total"] == 2
    assert summary["retry_count"] == 1
    assert summary["llm_retry_count"] == 1
    assert summary["retry_rate"] == 0.5
    assert summary["llm_retry_rate"] == 0.5
    assert summary["task_counter"] == Counter({"qa": 1, "summary": 1})


def test_analyze_eval_traces_summarizes_retry_modes_and_source_hits():
    analyze_traces = load_analyze_traces_module()
    records = [
        {
            "sample_id": "qa-tool-use-001",
            "mode": "hybrid",
            "task_type": "qa",
            "agent_debug": {
                "retry_triggered": True,
                "llm_retry_query": "tool use agent course concept",
            },
            "agent_eval": {
                "source_citation_hit": True,
            },
        },
        {
            "sample_id": "summary-tool-use-001",
            "mode": "hybrid",
            "task_type": "summary",
            "agent_debug": {
                "retry_triggered": False,
                "llm_retry_query": None,
            },
            "agent_eval": {
                "source_citation_hit": False,
            },
        },
    ]

    summary = analyze_traces.analyze_eval_traces(records)

    assert summary["total"] == 2
    assert summary["retry_count"] == 1
    assert summary["llm_retry_count"] == 1
    assert summary["retry_rate"] == 0.5
    assert summary["llm_retry_rate"] == 0.5
    assert summary["source_hit_true"] == 1
    assert summary["source_hit_total"] == 2
    assert summary["source_hit_rate"] == 0.5
    assert summary["mode_counter"] == Counter({"hybrid": 2})
    assert summary["task_counter"] == Counter({"qa": 1, "summary": 1})
    assert summary["retry_samples"] == Counter({"qa-tool-use-001": 1})
