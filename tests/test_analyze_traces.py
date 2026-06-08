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


def test_analyze_eval_failures_and_question_patterns_rank_unstable_items():
    analyze_traces = load_analyze_traces_module()
    records = [
        {
            "sample_id": "summary-tool-use-001",
            "mode": "hybrid",
            "task_type": "summary",
            "question": "帮我总结这节课",
            "agent_debug": {
                "retry_triggered": True,
                "llm_retry_query": "tool use lesson summary",
            },
            "agent_eval": {
                "source_citation_hit": False,
            },
        },
        {
            "sample_id": "summary-tool-use-001",
            "mode": "hybrid",
            "task_type": "summary",
            "question": "帮我总结这节课",
            "agent_debug": {
                "retry_triggered": True,
                "llm_retry_query": "tool use lesson summary",
            },
            "agent_eval": {
                "source_citation_hit": True,
            },
        },
        {
            "sample_id": "qa-tool-use-001",
            "mode": "hybrid",
            "task_type": "qa",
            "question": "tool use 是什么？",
            "agent_debug": {
                "retry_triggered": False,
                "llm_retry_query": None,
            },
            "agent_eval": {
                "source_citation_hit": True,
            },
        },
    ]

    failures = analyze_traces.analyze_eval_failures(records)
    patterns = analyze_traces.analyze_question_patterns(records)

    assert failures["retry_heavy_samples"][0]["sample_id"] == "summary-tool-use-001"
    assert failures["retry_heavy_samples"][0]["retry_rate"] == 1.0
    assert failures["llm_retry_heavy_samples"][0]["sample_id"] == "summary-tool-use-001"
    assert failures["source_miss_samples"][0]["sample_id"] == "summary-tool-use-001"
    assert patterns["question_retry_stats"][0]["question"] == "帮我总结这节课"
    assert patterns["question_retry_stats"][0]["retry_rate"] == 1.0


def test_generate_optimization_suggestions_returns_actionable_chinese_hints():
    analyze_traces = load_analyze_traces_module()

    cli_summary = {
        "total": 2,
        "retry_count": 0,
        "llm_retry_count": 0,
        "retry_rate": 0.0,
        "llm_retry_rate": 0.0,
        "task_counter": Counter({"qa": 2}),
    }
    eval_summary = {
        "total": 4,
        "retry_count": 2,
        "llm_retry_count": 2,
        "retry_rate": 0.5,
        "llm_retry_rate": 0.5,
        "source_hit_total": 4,
        "source_hit_true": 2,
        "source_hit_rate": 0.5,
        "mode_counter": Counter({"hybrid": 4}),
        "task_counter": Counter({"summary": 4}),
        "retry_samples": Counter({"summary-tool-use-001": 2}),
    }
    eval_failures = {
        "retry_heavy_samples": [
            {
                "sample_id": "summary-tool-use-001",
                "task_type": "summary",
                "retry_count": 2,
                "retry_rate": 1.0,
            }
        ],
        "llm_retry_heavy_samples": [
            {
                "sample_id": "summary-tool-use-001",
                "task_type": "summary",
                "llm_retry_count": 2,
                "llm_retry_rate": 1.0,
            }
        ],
        "source_miss_samples": [
            {
                "sample_id": "summary-tool-use-001",
                "task_type": "summary",
                "source_hit_false": 1,
                "source_miss_rate": 0.5,
            }
        ],
    }
    question_patterns = {
        "question_retry_stats": [
            {
                "question": "帮我总结这节课",
                "task_types": ["summary"],
                "retry_count": 2,
                "retry_rate": 1.0,
                "llm_retry_count": 2,
                "llm_retry_rate": 1.0,
            }
        ]
    }

    suggestions = analyze_traces.generate_optimization_suggestions(
        cli_summary,
        eval_summary,
        eval_failures,
        question_patterns,
    )

    assert any("总结类问题的 Retry 率偏高" in item for item in suggestions)
    assert any("LLM Retry 使用率偏高" in item for item in suggestions)
    assert any("来源引用命中率偏低" in item for item in suggestions)
    assert any("以下样本最常触发 Retry" in item for item in suggestions)
    assert any("评估集的 Retry 率明显高于 CLI 真实交互" in item for item in suggestions)


def test_analyze_task_type_breakdown_summarizes_retry_and_source_hits():
    analyze_traces = load_analyze_traces_module()
    records = [
        {
            "task_type": "summary",
            "agent_debug": {
                "retry_triggered": True,
                "llm_retry_query": "summary lesson rewrite",
            },
            "agent_eval": {
                "source_citation_hit": False,
            },
        },
        {
            "task_type": "summary",
            "agent_debug": {
                "retry_triggered": False,
                "llm_retry_query": None,
            },
            "agent_eval": {
                "source_citation_hit": True,
            },
        },
        {
            "task_type": "qa",
            "agent_debug": {
                "retry_triggered": False,
                "llm_retry_query": None,
            },
            "agent_eval": {
                "source_citation_hit": True,
            },
        },
    ]

    summaries = analyze_traces.analyze_task_type_breakdown(records)

    assert summaries["summary"]["total"] == 2
    assert summaries["summary"]["retry_count"] == 1
    assert summaries["summary"]["retry_rate"] == 0.5
    assert summaries["summary"]["llm_retry_count"] == 1
    assert summaries["summary"]["source_hit_true"] == 1
    assert summaries["summary"]["source_hit_total"] == 2
    assert summaries["summary"]["source_hit_rate"] == 0.5
    assert summaries["qa"]["total"] == 1
    assert summaries["qa"]["retry_rate"] == 0.0


def test_generate_task_type_suggestions_returns_task_specific_hints():
    analyze_traces = load_analyze_traces_module()
    task_summaries = {
        "summary": {
            "task_type": "summary",
            "total": 4,
            "retry_count": 3,
            "retry_rate": 0.75,
            "llm_retry_count": 2,
            "llm_retry_rate": 0.5,
            "source_hit_total": 4,
            "source_hit_true": 2,
            "source_hit_rate": 0.5,
        },
        "qa": {
            "task_type": "qa",
            "total": 3,
            "retry_count": 0,
            "retry_rate": 0.0,
            "llm_retry_count": 0,
            "llm_retry_rate": 0.0,
            "source_hit_total": 3,
            "source_hit_true": 3,
            "source_hit_rate": 1.0,
        },
    }

    suggestions = analyze_traces.generate_task_type_suggestions(task_summaries)

    assert any("summary 的 Retry 率偏高" in item for item in suggestions["summary"])
    assert any("summary 对 LLM rewrite 依赖偏高" in item for item in suggestions["summary"])
    assert any("summary 的来源引用命中率偏低" in item for item in suggestions["summary"])
    assert suggestions["qa"] == [
        "qa 当前没有暴露明显短板，可以先保持现状，继续扩大样本覆盖。"
    ]
