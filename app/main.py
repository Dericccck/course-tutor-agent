# main.py：跑程序
# 程序入口。先负责接收一个问题，然后调用 agent.py 返回结果。
from agent import ask_course_agent
from config import get_settings
from loader import load_documents


if __name__ == "__main__":
    settings = get_settings()
    documents = load_documents(settings.course_source_root)

    print(f"Loaded {len(documents)} documents")
    print("课程辅导 Agent 已启动。输入 exit / quit / q 退出。\n")

    while True:
        question = input("请输入你的问题：").strip()

        if question.lower() in {"exit", "quit", "q"}:
            print("已退出课程辅导 Agent。")
            break

        if not question:
            print("问题不能为空。\n")
            continue

        result = ask_course_agent(question, documents, settings=settings)

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
