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


def print_counter(title: str, counter: Counter) -> None:
    """打印 Counter 统计结果，按照出现频率从高到低排序。"""
    print(title)
    if not counter:
        print("  (empty)")
        return

    for key, value in counter.most_common():
        print(f"  - {key}: {value}")


def print_cli_report(summary: dict) -> None:
    print("\n=== CLI Trace Summary ===")
    print(f"Total: {summary['total']}")
    print(f"Retry Triggered: {summary['retry_count']} ({summary['retry_rate']:.2%})")
    print(f"LLM Retry Used: {summary['llm_retry_count']} ({summary['llm_retry_rate']:.2%})")
    print_counter("Task Types:", summary["task_counter"])


def print_eval_report(summary: dict) -> None:
    print("\n=== Eval Trace Summary ===")
    print(f"Total: {summary['total']}")
    print(f"Retry Triggered: {summary['retry_count']} ({summary['retry_rate']:.2%})")
    print(f"LLM Retry Used: {summary['llm_retry_count']} ({summary['llm_retry_rate']:.2%})")
    print(
        f"Source Citation Hit: {summary['source_hit_true']}/{summary['source_hit_total']} "
        f"({summary['source_hit_rate']:.2%})"
    )
    print_counter("Modes:", summary["mode_counter"])
    print_counter("Task Types:", summary["task_counter"])
    print_counter("Retry-heavy Samples:", summary["retry_samples"])


def main() -> None:
    """主函数，加载 CLI 和 Eval 的 trace 数据，进行分析，并打印报告。"""
    cli_records = load_jsonl(CLI_TRACE_PATH)
    eval_records = load_jsonl(EVAL_TRACE_PATH)

    print(f"CLI trace path: {CLI_TRACE_PATH}")
    print(f"Eval trace path: {EVAL_TRACE_PATH}")

    cli_summary = analyze_cli_traces(cli_records)
    eval_summary = analyze_eval_traces(eval_records)

    print_cli_report(cli_summary)
    print_eval_report(eval_summary)


if __name__ == "__main__":
    main()