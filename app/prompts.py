# prompts.py：组织输入
# 放系统提示词模板，比如“只能基于资料回答，不足就说不知道”。
from schemas import RetrievedChunk

SYSTEM_PROMPT = """你是一个课程辅导 Agent，负责基于本地课程资料回答用户问题。

你必须遵守以下规则：
1.只能基于提供的课程资料回答，不要凭空编造。
2.如果资料不足以支持结论，要明确说明“根据当前资料无法确定”。
3.回答尽量清晰、简介、适合学习场景。
4.优先解释概念，再给学习建议。
5.输出必须是JSON，且包含answer、suggestions、sources三个字段。
"""

def build_memory_block(memory: dict) -> str:
    goal = memory.get("learning_goal", "").strip()
    scope = memory.get("preferred_scope", "").strip()
    completed_topics = memory.get("completed_topics", [])

    lines: list[str] = []

    if goal:
        lines.append(f"学习目标：{goal}")

    if scope:
        lines.append(f"学习范围：{scope}")

    if completed_topics:
        lines.append("用户已完成的主题：")
        for topic in completed_topics:
            lines.append(f"- {topic}")

    if not lines:
        return "无"

    return "\n".join(lines)

# 因为检索结果本身是结构化的，但模型输入需要是文本。  结构化检索结果 -> 模型可读上下文
def build_context_block(retrieved_chunks: list[RetrievedChunk]) -> str:
    sections: list[str] = []

    for index, chunk in enumerate(retrieved_chunks, start=1):
        section = (
            f"[资料 {index}]\n"
            f"标题: {chunk.title}\n"
            f"来源: {chunk.source}\n"
            f"标签: {', '.join(chunk.tags) if chunk.tags else '无'}\n"
            f"相关片段: {chunk.snippet}\n"
        )
        sections.append(section)

    return "\n".join(sections)

# 只负责组织这一轮输入，不负责模型调用
def build_user_prompt(question: str, retrieved_chunks: list[RetrievedChunk], memory: dict | None = None,) -> str:
    context_block = build_context_block(retrieved_chunks)
    memory_block = build_memory_block(memory or {})

    return f"""用户问题：
    {question}

    用户当前记忆：
    {memory_block}
    
    相关课程资料：
    {context_block}

    请基于以上资料输出 JSON：
    {{
        "answer": "直接回答用户问题",
        "suggestions": ["给出1-3条后续学习建议"],
        "sources": ["列出你实际使用的资料来源路径"]
    }}
    """

def build_summary_prompt(question: str, retrieved_chunks: list[RetrievedChunk], memory: dict | None = None) -> str:
    content_block = build_context_block(retrieved_chunks)
    memory_block = build_memory_block(memory or {})

    return f"""用户希望你总结课程内容。

    用户请求：
    {question}

    用户当前记忆：
    {memory_block}

    相关课程资料：
    {content_block}

    请基于以上资料输出 JSON：
    {{
        "answer": "用清晰的方式总结这节课 / 这个notebook的核心内容",
        "suggestions": ["给出1-3条后续学习建议"],
        "sources": ["列出你实际使用的资料来源路径"]
    }}

    总结时尽量包括：
    1.这一节主要在讲什么
    2.他解决了什么问题
    3.它的关键收获是什么
    """

def build_study_plan_prompt(question: str, retrieved_chunks: list[RetrievedChunk], memory: dict | None = None,) -> str:
    context_block = build_context_block(retrieved_chunks)
    memory_block = build_memory_block(memory or {})

    completed_topics = memory.get("completed_topics", []) if memory else []
    remaining_titles = [chunk.title for chunk in retrieved_chunks if chunk.title not in completed_topics]
    completed_titles_in_context = [chunk.title for chunk in retrieved_chunks if chunk.title in completed_topics]
    remaining_titles_block = "\n".join(f"- {title}" for title in remaining_titles or "- 无")
    completed_titles_block = "\n".join(f"- {title}" for title in completed_titles_in_context or "- 无")

    allowed_titles = "\n".join(
        f"- {chunk.title}"
        for chunk in retrieved_chunks
    )

    return f"""用户希望你根据课程资料给出学习顺序建议。

    用户请求：
    {question}

    用户当前记忆：
    {memory_block}

    本次可优先推荐的未完成模块：
    {remaining_titles_block}

    本次上下文中已完成的模块：
    {completed_titles_block}

    本次允许引用的课程模块标题：
    {allowed_titles}

    相关课程资料：
    {context_block}

    请基于以上资料输出 JSON：
    {{
    "answer": "给出学习顺序建议，并解释为什么这样安排",
    "suggestions": ["给出 1-3 条后续学习建议"],
    "sources": ["列出你实际使用的资料来源路径"]
    }}

    回答时请尽量按这个结构组织：
    1. 第一阶段：先学什么
    2. 第二阶段：再学什么
    3. 第三阶段：最后学什么
    4. 每个阶段为什么这样安排
    5. 如果目标是做一个 AIAgent 项目，需要优先掌握哪些能力

    要求：
    - 优先结合课程目录之间的依赖关系安排顺序
    - 尽量引用具体课程模块或章节，而不是只写笼统的大目录
    - 如果用户目标是做一个 AIAgent 项目，学习路线应尽量同时覆盖：
    1. 基础能力
    2. RAG / 检索增强能力
    3. Agent 设计与工具调用能力
    - 只允许使用“本次允许引用的课程模块标题”中的真实标题名称
    - 不要把标题改写成你自己猜测的编号或缩写
    - 不要编造不存在的课程名、编号或目录结构
    - 如果无法确定具体编号，就直接使用上面的真实标题
    - 回答要像学习路线图，而不是泛泛推荐
    - 如果资料不足，不要编造不存在的课程内容
    - 对于“用户已完成的主题”，不要再把它们作为优先学习阶段重复推荐
    - 可以在路线中简短提到它们已完成，但后续建议应优先指向尚未完成的模块
    - 如果用户已经完成某个模块，应尽量推荐它的后续模块，而不是重复推荐同一模块
    - 如果“本次可优先推荐的未完成模块”不为空，优先从这些模块中安排后续学习路线
    - 不要把“本次上下文中已完成的模块”排成第一阶段或优先阶段，除非用户明确要求复习
    """