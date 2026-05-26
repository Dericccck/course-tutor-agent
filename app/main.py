# main.py：跑程序
# 程序入口。先负责接收一个问题，然后调用 agent.py 返回结果。
from schemas import DocumentChunk
from agent import ask_course_agent
from config import get_settings
from loader import load_documents, load_document_chunks
from memory import load_user_memory, save_user_memory, build_default_memory
from embedding_provider import build_embedding_provider
from vector_store import InMemoryVectorStore
from vector_index_cache import (
    build_vector_index_payload,
    is_vector_index_cache_valid,
    load_vector_index_cache,
    save_vector_index_cache,
)
from reranker import build_reranker


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
    print("- mark_done: 标记某个主题为已完成")
    print("- show_memory：查看当前学习记忆")
    print("- clear_goal：清除当前学习目标")
    print("- clear_scope：清除当前学习范围")
    print("- unmark_done: 主题名：取消某个已完成主题")
    print("- show_examples：查看示例问题")
    print("- show_progress：查看当前学习进度摘要")
    print("- clear_done：清空所有已完成主题")
    print("- reset_memory：重置全部学习记忆")
    print()

def print_progress(memory: dict) -> None:
    goal = memory.get("learning_goal", "").strip()
    scope = memory.get("preferred_scope", "").strip()
    completed_topics = memory.get("completed_topics", [])

    print("\n当前学习进度摘要：")
    print(f"- 学习目标: {goal or '未设置'}")
    print(f"- 学习范围: {scope or '未设置'}")
    print(f"- 已完成主题数量: {len(completed_topics)}")
    print(f"- 当前学习阶段: {get_learning_stage(memory)}")

    if completed_topics:
        print("- 已完成主题列表:")
        for topic in completed_topics:
            print(f"  - {topic}")
    else:
        print("- 已完成主题列表: 无")

    if completed_topics:
        print("- 当前建议: 优先继续学习尚未完成的后续模块。")
    else:
        print("- 当前建议: 可以先从基础模块开始建立整体框架。")
    
    next_topic = get_next_recommended_topic(memory)
    if next_topic:
        print(f"- 当前建议下一步: {next_topic}")
    else:
        print("- 当前建议下一步: 当前主学习路线已全部完成，可以开始复习或扩展新主题。")

    print()


def print_example_questions() -> None:
    print("示例问题：")
    print("- tool use 是什么，和 agent 有什么关系？")
    print("- 帮我总结 07-planning-design 这一节在讲什么")
    print("- 如果我只学 1-* 和 2-*，想做一个 AIAgent 项目，请按课程模块给我安排学习顺序。")
    print()

LEARNING_SEQUENCE = [
    "01 Intro To AI Agents 学习摘要",
    "02 Explore Agentic Frameworks 学习摘要",
    "03 Agentic Design Patterns 学习摘要",
    "05 Agentic RAG 学习摘要",
    "06 Building Trustworthy Agents 学习摘要",
    "11 Agentic Protocols 学习摘要",
]

def get_next_recommended_topic(memory: dict) -> str | None:
    completed_topics = memory.get("completed_topics", [])

    for topic in LEARNING_SEQUENCE:
        if topic not in completed_topics:
            return topic

    return None

def get_learning_stage(memory: dict) -> str:
    completed_topics = memory.get("completed_topics", [])

    if not completed_topics:
        return "基础起步阶段"

    if "01 Intro To AI Agents 学习摘要" in completed_topics and \
       "02 Explore Agentic Frameworks 学习摘要" not in completed_topics:
        return "基础理解阶段"

    if "02 Explore Agentic Frameworks 学习摘要" in completed_topics and \
       "03 Agentic Design Patterns 学习摘要" not in completed_topics:
        return "框架理解阶段"

    if "03 Agentic Design Patterns 学习摘要" in completed_topics and \
       "05 Agentic RAG 学习摘要" not in completed_topics:
        return "设计与增强阶段"

    return "深化与扩展阶段"


if __name__ == "__main__":
    settings = get_settings()
    documents = load_documents(settings.course_source_root, include_dirs=settings.course_include_dirs)
    chunks = load_document_chunks(settings.course_source_root, include_dirs=settings.course_include_dirs)
    print(f"Loaded {len(documents)} documents")
    print(f"Loaded {len(chunks)} chunks")
    
    vector_store = None
    reranker = None
    
    if settings.retrieval_mode in {"vector", "hybrid"}:
        embedding_provider = build_embedding_provider(settings.embedding_provider, settings.embedding_model_name, settings.embedding_cache_dir) # 根据配置构建对应的 embedding provider 实例
        vector_store = InMemoryVectorStore(embedding_provider)
        cache_loaded = False

        # 尝试加载向量索引缓存，如果缓存有效就直接用缓存来恢复向量索引，跳过 embedding 计算的过程，显著提升启动速度；如果缓存无效（比如配置发生了变化，或者 chunks 内容发生了变化），就正常走向量索引构建流程，构建完成后再把新的向量索引缓存写回本地文件。
        if settings.vector_index_cache_path:
            cache_payload = load_vector_index_cache(settings.vector_index_cache_path)

            if cache_payload and is_vector_index_cache_valid(cache_payload, settings, chunks): # 命中缓存 - 只有当缓存存在且有效时，才使用缓存来恢复向量索引。这样我们就能确保在配置或者数据发生变化时，能够正确地重建向量索引，而不是误用过期的缓存数据。
                print("加载到有效的向量索引缓存，正在使用缓存来恢复向量索引...")
                cached_chunks = [DocumentChunk(**item) for item in cache_payload["chunks"]]
                cached_embeddings = cache_payload["embeddings"]

                vector_store.load_index(cached_chunks, cached_embeddings) # 不再重新算 embedding，而是直接用缓存中的 chunks 和 embeddings 来恢复向量索引，这样能显著提升启动速度，尤其是当 chunks 数量较大时。
                cache_loaded = True
                print("向量索引已从缓存中恢复。")
        
        if not cache_loaded:
            print("没有找到有效的向量索引缓存，正在构建向量索引...")
            vector_store.index_chunks(chunks) # 自动重建
            print("向量索引构建完成。")
            if settings.vector_index_cache_path:
                cache_payload = build_vector_index_payload(settings, chunks, vector_store.embeddings)
                save_vector_index_cache(settings.vector_index_cache_path, cache_payload) # 自动覆盖缓存文件
                print("新的向量索引缓存已保存到本地文件。")
    
    if settings.retrieval_mode == "hybrid" and settings.reranker_provider != "none":
        reranker = build_reranker(
            settings.reranker_provider,
            settings.reranker_model_name,
            settings.reranker_cache_dir,
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
