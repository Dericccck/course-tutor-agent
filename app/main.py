# main.py：跑程序
# 程序入口。先负责接收一个问题，然后调用 agent.py 返回结果。
import os
import json
from pathlib import Path
from agent import ask_course_agent, detect_task_type
from config import get_settings
from loader import load_documents, load_document_chunks
from memory import load_user_memory, save_user_memory, build_default_memory, update_recent_focus
from reranker import build_reranker
from vector_index_service import build_vector_store_with_cache


def print_help() -> None:
    texts = load_cli_texts_config()
    help_texts = texts.get("help", {})

    print(help_texts.get("capabilities_title", "课程辅导 Agent 支持以下能力："))
    for line in help_texts.get("capabilities", []):
        print(line)
    print()
    print(help_texts.get("commands_title", "可用命令："))
    for line in help_texts.get("commands", []):
        print(line)
    print()

def print_progress(memory: dict) -> None:
    texts = load_cli_texts_config()
    progress_texts = texts.get("progress", {})
    goal = memory.get("learning_goal", "").strip()
    scope = memory.get("preferred_scope", "").strip()
    completed_topics = memory.get("completed_topics", [])

    print(f"\n{progress_texts.get('title', '当前学习进度摘要：')}")
    print(f"- 学习目标: {goal or progress_texts.get('goal_unset', '未设置')}")
    print(f"- 学习范围: {scope or progress_texts.get('scope_unset', '未设置')}")
    print(f"- 已完成主题数量: {len(completed_topics)}")
    print(f"- 当前学习阶段: {get_learning_stage(memory)}")

    if completed_topics:
        print(f"- {progress_texts.get('completed_topics_title', '已完成主题列表:')}")
        for topic in completed_topics:
            print(f"  - {topic}")
    else:
        print(
            f"- {progress_texts.get('completed_topics_title', '已完成主题列表:')} "
            f"{progress_texts.get('completed_topics_none', '无')}"
        )

    if completed_topics:
        print(
            f"- 当前建议: "
            f"{progress_texts.get('suggestion_when_in_progress', '优先继续学习尚未完成的后续模块。')}"
        )
    else:
        print(
            f"- 当前建议: "
            f"{progress_texts.get('suggestion_when_started', '可以先从基础模块开始建立整体框架。')}"
        )
    
    next_topic = get_next_recommended_topic(memory)
    next_topic_label = progress_texts.get("next_topic_label", "当前建议下一步")
    if next_topic:
        print(f"- {next_topic_label}: {next_topic}")
    else:
        print(
            f"- {next_topic_label}: "
            f"{progress_texts.get('next_topic_when_complete', '当前主学习路线已全部完成，可以开始复习或扩展新主题。')}"
        )

    print()

def should_show_retrieval_debug() -> bool:
    return os.getenv("SHOW_RETRIEVAL_DEBUG", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

def print_example_questions() -> None:
    texts = load_cli_texts_config()
    example_texts = texts.get("examples", {})
    print(example_texts.get("title", "示例问题："))
    for question in example_texts.get("questions", []):
        print(f"- {question}")
    print()

def load_study_plan_order_config() -> dict:
    with STUDY_PLAN_ORDER_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_cli_texts_config() -> dict:
    with CLI_TEXTS_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_learning_sequence() -> list[str]: # 从配置文件里加载学习顺序列表，这样我们就可以在不修改代码的情况下，灵活地调整学习顺序，或者根据不同的学习目标提供不同的学习顺序。配置文件中应该包含一个 default_title_order 字段，它是一个字符串列表，表示按照什么顺序来推荐学习课程模块。我们在推荐下一步学习主题时，就会按照这个顺序来检查哪些模块已经完成了，哪些模块还没有完成，从而给出合理的下一步建议。
    config = load_study_plan_order_config()
    return config.get("default_title_order", [])


def get_learning_stages() -> list[dict]:
    config = load_study_plan_order_config()
    return config.get("learning_stages", [])

STUDY_PLAN_ORDER_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "study_plan_order.json"
)
CLI_TEXTS_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "cli_texts.json"
)

def get_next_recommended_topic(memory: dict) -> str | None:
    completed_topics = memory.get("completed_topics", [])

    for topic in get_learning_sequence():
        if topic not in completed_topics:
            return topic

    return None

def build_recent_focus_from_question(question: str, task_type: str) -> str:
    question_text = question.strip()
    lowered = question_text.lower()

    if task_type == "summary":
        if "05-agentic-rag" in lowered or "agentic rag" in lowered:
            return "05 Agentic RAG 总结"
        if "04-tool-use" in lowered or "tool use" in lowered:
            return "04 Tool Use 总结"
        if "07-planning-design" in lowered or "planning" in lowered:
            return "07 Planning Design 总结"
        return "最近在复习课程总结内容"

    if task_type == "study_plan":
        if "rag" in lowered and "agentic rag" in lowered:
            return "RAG 到 Agentic RAG 学习路线"
        if "aiagent" in lowered or "agent 项目" in question_text:
            return "AIAgent 项目学习路线"
        return "最近在规划学习路线"

    if "tool use" in lowered:
        return "Tool use 与 agent tool calling"
    if "agentic rag" in lowered:
        return "Agentic RAG 基础概念"
    if "planning" in lowered:
        return "Planning agent 与任务拆解"
    if "memory" in lowered:
        return "Agent memory 基础概念"

    return "最近在学习课程问答主题"

def get_learning_stage(memory: dict) -> str:
    completed_topics = memory.get("completed_topics", [])
    learning_sequence = get_learning_sequence()
    completed_count = sum(
        1 for topic in learning_sequence
        if topic in completed_topics
    )
    learning_stages = get_learning_stages()

    for stage in learning_stages:
        if completed_count <= stage.get("max_completed", 999):
            return stage.get("label", "深化与扩展阶段")

    return "深化与扩展阶段"


if __name__ == "__main__":
    settings = get_settings()
    documents = load_documents(settings.course_source_root, include_dirs=settings.course_include_dirs)
    chunks = load_document_chunks(settings.course_source_root, include_dirs=settings.course_include_dirs)
    print(f"Loaded {len(documents)} documents")
    print(f"Loaded {len(chunks)} chunks")
    
    vector_store = None
    reranker = None
    
    if settings.retrieval.retrieval_mode in {"vector", "hybrid"}:
        vector_store = build_vector_store_with_cache(settings, chunks) # 构建一个带缓存机制的向量存储服务 - 这里我们把之前 main.py 里关于向量索引构建和缓存恢复的逻辑抽成了一个独立的服务函数 build_vector_store_with_cache，这样不仅让 main.py 的代码更简洁清晰，也让这个向量索引构建和缓存恢复的功能变得更可复用，在其他地方如果需要类似的功能时，就可以直接调用这个服务函数，而不需要重复编写相同的逻辑。这个服务函数会根据当前的配置和数据状态，智能地决定是直接加载缓存来恢复向量索引，还是重新计算 embeddings 来构建向量索引，并且在构建完成后自动更新缓存文件，这样我们就能在保证数据一致性的前提下，显著提升程序的启动速度，尤其是在 chunks 数量较大时。
    
    if settings.retrieval.retrieval_mode == "hybrid" and settings.retrieval.reranker_provider != "none":
        reranker = build_reranker(
            settings.retrieval.reranker_provider,
            settings.retrieval.reranker_model_name,
            settings.retrieval.reranker_cache_dir,
        )
    
    memory = load_user_memory()

    print("课程辅导 Agent 已启动。\n")
    print_help()
    print_example_questions()

    turn = 1
    while True:
        question = input(f"[问题 {turn}] 请输入你的问题：").strip()

        if question.lower() in {"exit", "quit", "q"}:
            print("已退出课程辅导 Agent。")
            break

        if question.lower() == "help":
            print()
            print_help()
            continue

        if question.startswith("set_goal:"):
            goal = question.removeprefix("set_goal:").strip()

            if not goal:
                print("学习目标不能为空。\n")
                continue

            memory["learning_goal"] = goal
            save_user_memory(memory)

            print(f"学习目标已保存：{goal}\n")
            continue

        if question.startswith("set_scope:"):
            scope = question.removeprefix("set_scope:").strip()

            if not scope:
                print("学习范围不能为空。\n")
                continue

            memory["preferred_scope"] = scope
            save_user_memory(memory)

            print(f"学习范围已保存：{scope}\n")
            continue

        if question.startswith("mark_done:"):
            topic = question.removeprefix("mark_done:").strip()

            if not topic:
                print("已完成主题不能为空。\n")

            completed_topics = memory.setdefault("completed_topics", [])

            if topic in completed_topics:
                print(f"该主题已存在：{topic}\n")
            else:
                completed_topics.append(topic)
                save_user_memory(memory)
                print(f"已完成主题已保存：{topic}\n")
            continue

        if question == "show_memory":
            print("\n当前学习记忆：")
            print(f"- 学习目标: {memory.get('learning_goal', '') or '未设置'}")
            print(f"- 学习范围: {memory.get('preferred_scope', '') or '未设置'}")

            completed_topics = memory.get("completed_topics", [])
            if completed_topics:
                print("- 已完成主题:")
                for topic in completed_topics:
                    print(f"  - {topic}")
            else:
                print("- 已完成主题: 无")

            print()
            continue

        if question == "show_progress":
            print_progress(memory)
            continue

        if question == "clear_done":
            memory["completed_topics"] = []
            save_user_memory(memory)
            print("已完成主题已清空。\n")
            continue

        if question == "reset_memory":
            memory = build_default_memory()
            save_user_memory(memory)
            print("学习记忆已重置。\n")
            continue

        if question == "clear_goal":
            memory["learning_goal"] = ""
            save_user_memory(memory)
            print("学习目标已清除。\n")
            continue

        if question == "clear_scope":
            memory["preferred_scope"] = ""
            save_user_memory(memory)
            print("学习范围已清除。\n")
            continue

        if question.startswith("unmark_done:"):
            topic = question.removeprefix("unmark_done:").strip()
            if not topic:
                print("要取消的主题不能为空。\n")
                continue
            completed_topics = memory.get("completed_topics", [])
            if topic not in completed_topics:
                print(f"未找到该已完成主题：{topic}\n")
            else:
                completed_topics.remove(topic)
                save_user_memory(memory)
                print(f"已取消完成主题：{topic}\n")
            continue

        if question == "show_examples":
            print()
            print_example_questions()
            continue

        if not question:
            print("问题不能为空。\n")
            continue

        result = ask_course_agent(question, documents, settings=settings, memory=memory, chunks=chunks, vector_store=vector_store, reranker=reranker,)

        # 记忆系统保存“最近在学什么”
        task_type = detect_task_type(question)
        recent_focus = build_recent_focus_from_question(question, task_type)
        update_recent_focus(memory, recent_focus)
        save_user_memory(memory)

        print(f"\nQuestion: {question}\n")
        print("Answer:")
        print(result.answer)

        print("\nSuggestions:")
        for suggestion in result.suggestions:
            print(f"- {suggestion}")

        print("\nSources:")
        for source in result.sources:
            print(f"- {source}")

        if should_show_retrieval_debug() and result.debug:
            print("Debug:")
            print(f"- Task Type: {result.debug.get('task_type')}")
            print(f"- Initial Queries: {result.debug.get('initial_queries')}")
            print(f"- Retry Triggered: {result.debug.get('retry_triggered')}")
            print(f"- Retry Queries: {result.debug.get('retry_queries')}")
            print(f"- Initial Result Count: {result.debug.get('initial_result_count')}")
            print(f"- Final Result Count: {result.debug.get('final_result_count')}")
            print()

        print("\n" + "=" * 60 + "\n")
        
        save_user_memory(memory)
        turn += 1
