# trace analysis 脚本

# 回答几个现在非常关键的问题：
    # 最近是不是太频繁触发 retry 了？
    # llm_retry_query 到底有没有被用上？
    # 哪类任务最容易触发 retry？
    # 哪些 eval sample 经常不稳？
    # Source Citation Hit 在 trace 层面是否还能对上？



import json
# Counter 是一个非常有用的工具，可以帮助我们快速统计和分析数据中的频率分布。
# 在这个脚本中，我们使用 Counter 来统计不同任务类型、模式，以及重试情况的出现次数，从而更好地理解模型在不同场景下的表现和行为模式。
# 通过这些统计信息，我们可以识别出哪些任务类型或模式更容易触发重试，或者哪些样本更具有挑战性，从而为后续的优化提供有针对性的方向。
from collections import Counter 
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLI_TRACE_PATH = PROJECT_ROOT / "data" / "retrieval_traces.jsonl"
EVAL_TRACE_PATH = PROJECT_ROOT / "eval" / "eval_traces.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    """加载 JSONL 文件，每行一个 JSON 对象，返回一个字典列表。
    这个函数会跳过空行，并且在读取每行时会去除前后空白字符，确保我们得到的都是有效的 JSON 数据。
    通过使用这个函数，我们可以方便地加载和处理模型运行过程中生成的 trace 数据，从而进行更深入的分析和理解模型的行为。"""
    if not path.exists():
        return []

    items: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            items.append(json.loads(text))
    return items

def group_records(records: list[dict], key_name: str) -> dict[str, list[dict]]:
    """按某个字段把 trace 记录分组，方便后面做样本级统计。"""
    grouped: dict[str, list[dict]] = {}

    for item in records:
        key = item.get(key_name, "unknown")
        grouped.setdefault(key, []).append(item)

    return grouped


def analyze_cli_traces(records: list[dict]) -> dict:
    """分析 CLI trace 的记录，统计不同任务类型的出现次数，以及重试的情况。
    这个分析可以帮助我们了解模型在实际使用中的表现，以及哪些任务类型更容易触发重试，从而为后续的优化提供数据支持。"""
    task_counter = Counter()
    retry_count = 0
    llm_retry_count = 0

    for item in records:
        debug = item.get("debug", {})
        task_type = debug.get("task_type", "unknown")
        task_counter[task_type] += 1

        if debug.get("retry_triggered"):
            retry_count += 1

        if debug.get("llm_retry_query"):
            llm_retry_count += 1

    total = len(records)
    return {
        "total": total,
        "task_counter": task_counter,
        "retry_count": retry_count,
        "llm_retry_count": llm_retry_count,
        "retry_rate": (retry_count / total) if total else 0.0,
        "llm_retry_rate": (llm_retry_count / total) if total else 0.0,
    }


def analyze_eval_traces(records: list[dict]) -> dict:
    """分析 Eval trace 的记录，统计不同模式和任务类型的出现次数，以及重试和来源命中情况。
    通过这个分析，我们可以更全面地了解模型在评测环境中的表现，以及哪些模式或任务类型更具有挑战性，从而为后续的优化提供有针对性的数据支持。
    这个分析还可以帮助我们识别出哪些样本更容易触发重试，或者哪些样本的检索结果更容易被模型正确引用，从而指导我们在模型训练和检索策略优化时，重点关注这些具有挑战性的样本。"""
    mode_counter = Counter()
    task_counter = Counter()
    retry_count = 0
    llm_retry_count = 0
    source_hit_total = 0
    source_hit_true = 0
    retry_samples = Counter()

    for item in records:
        mode_counter[item.get("mode", "unknown")] += 1
        task_counter[item.get("task_type", "unknown")] += 1

        agent_debug = item.get("agent_debug", {})
        if agent_debug.get("retry_triggered"):
            retry_count += 1
            sample_id = item.get("sample_id", "unknown")
            retry_samples[sample_id] += 1

        if agent_debug.get("llm_retry_query"):
            llm_retry_count += 1

        agent_eval = item.get("agent_eval")
        if agent_eval and agent_eval.get("source_citation_hit") is not None:
            source_hit_total += 1
            if agent_eval.get("source_citation_hit"):
                source_hit_true += 1

    total = len(records)
    return {
        "total": total,
        "mode_counter": mode_counter,
        "task_counter": task_counter,
        "retry_count": retry_count,
        "llm_retry_count": llm_retry_count,
        "retry_rate": (retry_count / total) if total else 0.0,
        "llm_retry_rate": (llm_retry_count / total) if total else 0.0,
        "source_hit_total": source_hit_total,
        "source_hit_true": source_hit_true,
        "source_hit_rate": (source_hit_true / source_hit_total) if source_hit_total else 0.0,
        "retry_samples": retry_samples,
    }

def analyze_eval_failures(records: list[dict]) -> dict:
    """对 eval traces 做样本级失败分析，找出最不稳定的样本。"""
    grouped = group_records(records, "sample_id")

    retry_heavy_samples: list[dict] = []
    source_miss_samples: list[dict] = []
    llm_retry_heavy_samples: list[dict] = []

    for sample_id, items in grouped.items():
        total = len(items)
        if total == 0:
            continue

        retry_count = 0
        llm_retry_count = 0
        source_hit_total = 0
        source_hit_false = 0
        question = items[0].get("question", "")
        task_type = items[0].get("task_type", "unknown")

        for item in items:
            agent_debug = item.get("agent_debug", {})
            if agent_debug.get("retry_triggered"):
                retry_count += 1

            if agent_debug.get("llm_retry_query"):
                llm_retry_count += 1

            agent_eval = item.get("agent_eval")
            if agent_eval and agent_eval.get("source_citation_hit") is not None:
                source_hit_total += 1
                if not agent_eval.get("source_citation_hit"):
                    source_hit_false += 1

        retry_heavy_samples.append(
            {
                "sample_id": sample_id,
                "question": question,
                "task_type": task_type,
                "total": total,
                "retry_count": retry_count,
                "retry_rate": retry_count / total,
            }
        )

        llm_retry_heavy_samples.append(
            {
                "sample_id": sample_id,
                "question": question,
                "task_type": task_type,
                "total": total,
                "llm_retry_count": llm_retry_count,
                "llm_retry_rate": llm_retry_count / total,
            }
        )

        if source_hit_total > 0:
            source_miss_samples.append(
                {
                    "sample_id": sample_id,
                    "question": question,
                    "task_type": task_type,
                    "source_hit_total": source_hit_total,
                    "source_hit_false": source_hit_false,
                    "source_miss_rate": source_hit_false / source_hit_total,
                }
            )

    retry_heavy_samples.sort(key=lambda item: (-item["retry_rate"], -item["retry_count"], item["sample_id"]))
    llm_retry_heavy_samples.sort(key=lambda item: (-item["llm_retry_rate"], -item["llm_retry_count"], item["sample_id"]))
    source_miss_samples.sort(key=lambda item: (-item["source_miss_rate"], -item["source_hit_false"], item["sample_id"]))

    return {
        "retry_heavy_samples": retry_heavy_samples,
        "llm_retry_heavy_samples": llm_retry_heavy_samples,
        "source_miss_samples": source_miss_samples,
    }

def analyze_question_patterns(records: list[dict]) -> dict:
    """按 question 文本聚合，观察哪些问法更容易触发 retry 或 citation miss。"""
    grouped = group_records(records, "question")

    question_retry_stats: list[dict] = []

    for question, items in grouped.items():
        total = len(items)
        if total == 0:
            continue

        retry_count = 0
        llm_retry_count = 0
        task_types = set()

        for item in items:
            task_types.add(item.get("task_type", "unknown"))
            agent_debug = item.get("agent_debug", {})
            if agent_debug.get("retry_triggered"):
                retry_count += 1
            if agent_debug.get("llm_retry_query"):
                llm_retry_count += 1

        question_retry_stats.append(
            {
                "question": question,
                "task_types": sorted(task_types),
                "total": total,
                "retry_count": retry_count,
                "retry_rate": retry_count / total,
                "llm_retry_count": llm_retry_count,
                "llm_retry_rate": llm_retry_count / total,
            }
        )

    question_retry_stats.sort(
        key=lambda item: (-item["retry_rate"], -item["llm_retry_rate"], item["question"])
    )

    return {
        "question_retry_stats": question_retry_stats,
    }

def analyze_task_type_breakdown(records: list[dict]) -> dict:
    """按 task_type 聚合，观察不同任务类型的 retry / llm rewrite / source hit 表现。"""
    grouped = group_records(records, "task_type")
    task_summaries: dict[str, dict] = {}

    for task_type, items in grouped.items():
        total = len(items)
        retry_count = 0
        llm_retry_count = 0
        source_hit_total = 0
        source_hit_true = 0

        for item in items:
            agent_debug = item.get("agent_debug", {})
            if agent_debug.get("retry_triggered"):
                retry_count += 1
            if agent_debug.get("llm_retry_query"):
                llm_retry_count += 1

            agent_eval = item.get("agent_eval")
            if agent_eval and agent_eval.get("source_citation_hit") is not None:
                source_hit_total += 1
                if agent_eval.get("source_citation_hit"):
                    source_hit_true += 1

        task_summaries[task_type] = {
            "task_type": task_type,
            "total": total,
            "retry_count": retry_count,
            "retry_rate": (retry_count / total) if total else 0.0,
            "llm_retry_count": llm_retry_count,
            "llm_retry_rate": (llm_retry_count / total) if total else 0.0,
            "source_hit_total": source_hit_total,
            "source_hit_true": source_hit_true,
            "source_hit_rate": (source_hit_true / source_hit_total) if source_hit_total else 0.0,
        }

    return task_summaries

def generate_optimization_suggestions(
    cli_summary: dict,
    eval_summary: dict,
    eval_failures: dict,
    question_patterns: dict,
) -> list[str]:
    """根据 trace 统计结果生成下一步优化建议。"""
    suggestions: list[str] = []

    # 1. summary 类问题如果 retry 很高，优先优化 generic summary 的首轮 query
    summary_retry_candidates = [
        item for item in question_patterns["question_retry_stats"]
        if "summary" in item.get("task_types", []) and item["retry_rate"] >= 0.5
    ]
    if summary_retry_candidates:
        suggestions.append(
            "总结类问题的 Retry 率偏高，优先优化 generic summary 的首轮 query，减少“帮我总结这节课”这类问题对二轮补强的依赖。"
        )

    # 2. 如果 LLM retry 用得很多，说明规则型 query 已经不够，需要重点检查 rewrite 质量
    if eval_summary["llm_retry_rate"] >= 0.3:
        suggestions.append(
            "LLM Retry 使用率偏高，建议优先检查 rewrite 生成的 query 质量，而不是继续扩大 Retry 触发范围。"
        )

    # 3. 如果来源命中率不高，优先看 grounding / citation
    if eval_summary["source_hit_total"] > 0 and eval_summary["source_hit_rate"] < 0.8:
        suggestions.append(
            "来源引用命中率偏低，建议优先检查 grounding prompt、source normalization，以及 summary / qa 的 source 约束。"
        )

    # 4. 如果某些样本反复 retry，应该优先拿这些样本做定点优化
    retry_heavy_samples = eval_failures["retry_heavy_samples"][:3]
    if retry_heavy_samples and retry_heavy_samples[0]["retry_rate"] >= 0.5:
        sample_ids = ", ".join(item["sample_id"] for item in retry_heavy_samples)
        suggestions.append(
            f"以下样本最常触发 Retry：{sample_ids}，建议优先做定点分析，而不是继续全局加规则。"
        )

    # 5. 如果 CLI 侧 retry 很低但 eval 侧 retry 高，说明真实交互和评估集分布差异大
    if cli_summary["total"] > 0 and eval_summary["total"] > 0:
        if eval_summary["retry_rate"] - cli_summary["retry_rate"] >= 0.2:
            suggestions.append(
                "评估集的 Retry 率明显高于 CLI 真实交互，建议检查 eval 样本是否更偏难题或泛问题，并针对评估集补检索策略。"
            )

    if not suggestions:
        suggestions.append("当前 trace 没有暴露明显短板，下一步可以继续扩大评估覆盖或开始接 tracing 平台。")

    return suggestions

def generate_task_type_suggestions(task_summaries: dict[str, dict]) -> dict[str, list[str]]:
    """针对 qa / summary / study_plan 分别生成更具体的优化建议。"""
    suggestions_by_task: dict[str, list[str]] = {}

    for task_type, summary in task_summaries.items():
        suggestions: list[str] = []

        retry_rate = summary["retry_rate"]
        llm_retry_rate = summary["llm_retry_rate"]
        source_hit_total = summary["source_hit_total"]
        source_hit_rate = summary["source_hit_rate"]

        if task_type == "summary":
            if retry_rate >= 0.4:
                suggestions.append("summary 的 Retry 率偏高，优先优化 generic summary 的首轮 query 和课程锚点识别。")
            if llm_retry_rate >= 0.3:
                suggestions.append("summary 对 LLM rewrite 依赖偏高，建议检查 rewrite 是否真正提供了 lesson / notebook / 章节锚点。")
            if source_hit_total > 0 and source_hit_rate < 0.8:
                suggestions.append("summary 的来源引用命中率偏低，建议优先检查总结 prompt 的 source 约束和 summary narrowing。")

        elif task_type == "qa":
            if retry_rate >= 0.4:
                suggestions.append("qa 的 Retry 率偏高，建议优先优化概念题的首轮 query，而不是继续扩大 retry 范围。")
            if llm_retry_rate >= 0.3:
                suggestions.append("qa 对 LLM rewrite 依赖偏高，建议检查规则型 query 是否已经覆盖常见概念词。")
            if source_hit_total > 0 and source_hit_rate < 0.8:
                suggestions.append("qa 的来源引用命中率偏低，建议优先检查 grounding prompt 和 sources 归一化逻辑。")

        elif task_type == "study_plan":
            if retry_rate >= 0.4:
                suggestions.append("study_plan 的 Retry 率偏高，建议优先优化学习路线类问题的首轮 query，以及 goal / recent_focus 的利用方式。")
            if llm_retry_rate >= 0.3:
                suggestions.append("study_plan 对 LLM rewrite 依赖偏高，建议检查规则型 roadmap query 是否已经足够覆盖课程主线。")
            if source_hit_total > 0 and source_hit_rate < 0.8:
                suggestions.append("study_plan 的来源引用命中率偏低，建议优先检查 post-rank 后的 prompt_chunks 是否仍然保留了关键路径来源。")

        if not suggestions:
            suggestions.append(f"{task_type} 当前没有暴露明显短板，可以先保持现状，继续扩大样本覆盖。")

        suggestions_by_task[task_type] = suggestions

    return suggestions_by_task

def print_task_type_suggestions(task_suggestions: dict[str, list[str]]) -> None:
    print("\n=== 按任务类型拆分的优化建议 ===")

    if not task_suggestions:
        print("（空）")
        return

    for task_type, suggestions in task_suggestions.items():
        print(f"\n[{task_type}]")
        for item in suggestions:
            print(f"- {item}")


def print_counter(title: str, counter: Counter) -> None:
    """打印 Counter 统计结果，按照出现频率从高到低排序。"""
    print(title)
    if not counter:
        print("  （空）")
        return

    for key, value in counter.most_common():
        print(f"  - {key}: {value}")

def print_top_items(title: str, items: list[dict], fields: list[str], top_n: int = 5) -> None:
    """打印前 N 个高风险/高频样本，避免一次输出太长。"""
    print(title)
    if not items:
        print("  （空）")
        return

    for item in items[:top_n]:
        values = [f"{field}={item.get(field)}" for field in fields]
        print("  - " + ", ".join(values))


def print_cli_report(summary: dict) -> None:
    print("\n=== CLI Trace 汇总 ===")
    print(f"总条数: {summary['total']}")
    print(f"触发 Retry: {summary['retry_count']} ({summary['retry_rate']:.2%})")
    print(f"使用 LLM Retry: {summary['llm_retry_count']} ({summary['llm_retry_rate']:.2%})")
    print_counter("任务类型分布:", summary["task_counter"])


def print_eval_report(summary: dict, failures: dict, patterns: dict) -> None:
    print("\n=== Eval Trace 汇总 ===")
    print(f"总条数: {summary['total']}")
    print(f"触发 Retry: {summary['retry_count']} ({summary['retry_rate']:.2%})")
    print(f"使用 LLM Retry: {summary['llm_retry_count']} ({summary['llm_retry_rate']:.2%})")
    print(
        f"来源引用命中: {summary['source_hit_true']}/{summary['source_hit_total']} "
        f"({summary['source_hit_rate']:.2%})"
    )
    print_counter("模式分布:", summary["mode_counter"])
    print_counter("任务类型分布:", summary["task_counter"])
    print_counter("高频 Retry 样本:", summary["retry_samples"])
    print_top_items(
        "最容易触发 Retry 的样本:",
        failures["retry_heavy_samples"],
        ["sample_id", "task_type", "retry_count", "retry_rate"],
    )
    print_top_items(
        "最依赖 LLM Retry 的样本:",
        failures["llm_retry_heavy_samples"],
        ["sample_id", "task_type", "llm_retry_count", "llm_retry_rate"],
    )
    print_top_items(
        "来源引用命中不稳的样本:",
        failures["source_miss_samples"],
        ["sample_id", "task_type", "source_hit_false", "source_miss_rate"],
    )
    print_top_items(
        "最容易触发 Retry 的问题:",
        patterns["question_retry_stats"],
        ["question", "task_types", "retry_count", "retry_rate", "llm_retry_count"],
    )

def print_optimization_suggestions(suggestions: list[str]) -> None:
    print("\n=== 下一步优化建议 ===")
    for item in suggestions:
        print(f"- {item}")

def print_task_type_summary(task_summaries: dict[str, dict]) -> None:
    print("\n=== 按任务类型拆分的统计 ===")

    if not task_summaries:
        print("（空）")
        return

    for task_type, summary in sorted(task_summaries.items()):
        print(f"\n[{task_type}]")
        print(f"- 样本数: {summary['total']}")
        print(f"- Retry: {summary['retry_count']} ({summary['retry_rate']:.2%})")
        print(f"- LLM Retry: {summary['llm_retry_count']} ({summary['llm_retry_rate']:.2%})")
        print(
            f"- 来源引用命中: {summary['source_hit_true']}/{summary['source_hit_total']} "
            f"({summary['source_hit_rate']:.2%})"
        )

def main() -> None:
    """主函数，加载 CLI 和 Eval 的 trace 数据，进行分析，并打印报告。"""
    cli_records = load_jsonl(CLI_TRACE_PATH)
    eval_records = load_jsonl(EVAL_TRACE_PATH)

    print(f"CLI trace 路径: {CLI_TRACE_PATH}")
    print(f"Eval trace 路径: {EVAL_TRACE_PATH}")

    cli_summary = analyze_cli_traces(cli_records)
    eval_summary = analyze_eval_traces(eval_records)
    eval_failures = analyze_eval_failures(eval_records)
    question_patterns = analyze_question_patterns(eval_records)
    task_summaries = analyze_task_type_breakdown(eval_records)
    suggestions = generate_optimization_suggestions(
        cli_summary,
        eval_summary,
        eval_failures,
        question_patterns,
    )
    task_suggestions = generate_task_type_suggestions(task_summaries)

    print_cli_report(cli_summary)
    print_eval_report(eval_summary, eval_failures, question_patterns)
    print_task_type_summary(task_summaries)
    print_optimization_suggestions(suggestions)
    print_task_type_suggestions(task_suggestions)


if __name__ == "__main__":
    main()
