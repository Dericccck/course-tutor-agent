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
def build_user_prompt(question: str, retrieved_chunks: list[RetrievedChunk]) -> str:
    context_block = build_context_block(retrieved_chunks)

    return f"""用户问题：
    {question}
    
    相关课程资料：
    {context_block}

    请基于以上资料输出 JSON：
    {{
        "answer": "直接回答用户问题",
        "suggestions": ["给出1-3条后续学习建议"],
        "sources": ["列出你实际使用的资料来源路径"]
    }}
    """

def build_summary_prompt(question: str, retrieved_chunks: list[RetrievedChunk]) -> str:
    content_block = build_context_block(retrieved_chunks)

    return f"""用户希望你总结课程内容。

    用户请求：
    {question}

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