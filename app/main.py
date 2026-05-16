# main.py：跑程序
# 程序入口。先负责接收一个问题，然后调用 agent.py 返回结果。
from agent import ask_course_agent
from config import get_settings
from loader import load_documents
from memory import load_user_memory, save_user_memory


def print_help() -> None:
    print("课程辅导 Agent 支持以下能力：")
    print("1. 普通课程问答")
    print("2. 某节课 / 某个 notebook 总结")
    print("3. 根据学习目标生成学习顺序建议")
    print()
    print("可用命令：")
    print("- 直接输入问题：开始提问")
    print("- help：查看帮助")
    print("- set_goal: 你的学习目标")
    print("- set_scope: 你的学习范围")
    print("- exit / quit / q：退出程序")
    print()


def print_example_questions() -> None:
    print("示例问题：")
    print("- tool use 是什么，和 agent 有什么关系？")
    print("- 帮我总结 07-planning-design 这一节在讲什么")
    print("- 如果我只学 1-* 和 2-*，想做一个 AIAgent 项目，请按课程模块给我安排学习顺序。")
    print()


if __name__ == "__main__":
    settings = get_settings()
    documents = load_documents(settings.course_source_root)
    memory = load_user_memory()

    print(f"Loaded {len(documents)} documents")
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

        if not question:
            print("问题不能为空。\n")
            continue

        result = ask_course_agent(question, documents, settings=settings, memory=memory)

        print(f"\nQuestion: {question}\n")
        print("Answer:")
        print(result.answer)

        print("\nSuggestions:")
        for suggestion in result.suggestions:
            print(f"- {suggestion}")

        print("\nSources:")
        for source in result.sources:
            print(f"- {source}")

        print("\n" + "=" * 60 + "\n")
        
        save_user_memory(memory)
        turn += 1
