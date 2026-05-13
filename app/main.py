# main.py：跑程序
# 程序入口。先负责接收一个问题，然后调用 agent.py 返回结果。
from loader import load_documents
from retriever import retrieve_documents
from prompts import build_user_prompt
from agent import ask_course_agent
from config import get_settings

if __name__ == "__main__":
    settings = get_settings()
    documents = load_documents(settings.course_source_root)

    print(f"Loaded {len(documents)} documents")

    # question = "tool use 是什么， 和 agent 有什么关系？"
    question = input("请输入你的问题：").strip()
    
    if not question:
        print("问题不能为空。")
    else:
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
