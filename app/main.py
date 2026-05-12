# 程序入口。先负责接收一个问题，然后调用 agent.py 返回结果。
from loader import load_documents
from retriever import retrieve_documents

if __name__ == "__main__":
    root_dir = "/Users/a1-6/Desktop/AIAgent/code"
    documents = load_documents(root_dir)

    print(f"Loaded {len(documents)} documents")

    query = "tool use 是什么， 和 agent 有什么关系？"
    results = retrieve_documents(query, documents, top_k=5)

    print(f"\nQuery: {query}\n")
    print(f"Retrieved {len(results)} results\n")

    for item in results:
        print("-"* 60)
        print(f"title: {item.title}")
        print(f"score: {item.score}")
        print(f"source: {item.source}")
        print(f"tags: {item.tags}")
        print(f"snippet: {item.snippet}")