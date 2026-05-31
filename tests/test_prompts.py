# 测试 prompt 组织逻辑是否符合当前项目预期：
# 1. memory 为空时应返回“无”
# 2. memory 有值时应正确展开目标、范围和已完成主题
# 3. 普通问答 prompt 应包含问题、记忆和资料片段
# 4. 学习路线 prompt 应区分未完成模块和已完成模块

from prompts import (
    build_context_block,
    build_memory_block,
    build_study_plan_prompt,
    build_user_prompt,
)
from schemas import RetrievedChunk


def make_chunk(title: str, source: str, chunk_id: str | None = None) -> RetrievedChunk:
    # 构造一个最小可用的检索结果对象，供 prompt 测试使用
    return RetrievedChunk(
        source=source,
        title=title,
        chunk_id=chunk_id,
        snippet=f"{title} 的相关片段",
        score=10.0,
        tags=["agent"],
    )


def test_build_memory_block_returns_none_text_when_empty():
    # 当 memory 为空时，应返回“无”
    result = build_memory_block({})
    assert result == "无"


def test_build_memory_block_renders_memory_fields():
    # 当 memory 有内容时，应正确渲染学习目标、范围和已完成主题
    memory = {
        "learning_goal": "我想做一个 AIAgent 项目",
        "preferred_scope": "我只学习 1-* 和 2-* 的内容",
        "completed_topics": ["01 Intro To AI Agents 学习摘要"],
    }

    result = build_memory_block(memory)

    assert "学习目标" in result
    assert "学习范围" in result
    assert "用户已完成的主题" in result
    assert "01 Intro To AI Agents 学习摘要" in result


def test_build_user_prompt_contains_question_memory_and_context():
    # 普通问答 prompt 应同时包含问题、用户记忆和资料上下文
    memory = {
        "learning_goal": "我想做一个 AIAgent 项目",
        "preferred_scope": "我只学习 1-* 和 2-* 的内容",
        "completed_topics": [],
    }
    chunks = [
        make_chunk(
            "04 Tool Use 学习摘要",
            "/tmp/04-tool-use/notebook-summary.md",
        )
    ]

    prompt = build_user_prompt("tool use 是什么？", chunks, memory=memory)

    assert "tool use 是什么？" in prompt
    assert "用户当前记忆" in prompt
    assert "学习目标" in prompt
    assert "04 Tool Use 学习摘要" in prompt
    assert "优先基于前 3 条资料回答" in prompt
    assert "列出你实际使用的资料来源路径" in prompt


def test_build_study_plan_prompt_splits_remaining_and_completed_titles():
    # 学习路线 prompt 应区分未完成模块与已完成模块
    memory = {
        "learning_goal": "我想做一个 AIAgent 项目",
        "preferred_scope": "我只学习 1-* 和 2-* 的内容",
        "completed_topics": ["01 Intro To AI Agents 学习摘要"],
    }
    chunks = [
        make_chunk(
            "01 Intro To AI Agents 学习摘要",
            "/tmp/01-intro-to-ai-agents/notebook-summary.md",
        ),
        make_chunk(
            "02 Explore Agentic Frameworks 学习摘要",
            "/tmp/02-explore-agentic-frameworks/notebook-summary.md",
        ),
    ]

    prompt = build_study_plan_prompt(
        "如果我想继续做一个 AIAgent 项目，接下来应该按课程模块怎么学？",
        chunks,
        memory=memory,
    )

    assert "本次可优先推荐的未完成模块" in prompt
    assert "本次上下文中已完成的模块" in prompt
    assert "- 02 Explore Agentic Frameworks 学习摘要" in prompt
    assert "- 01 Intro To AI Agents 学习摘要" in prompt
    assert "优先基于前 5 条资料安排学习路线" in prompt
    assert "不要把“本次上下文中已完成的模块”排成第一阶段或优先阶段" in prompt


def test_build_context_block_renders_chunk_level_source_reference():
    chunks = [
        make_chunk(
            "04 Tool Use 学习摘要",
            "/tmp/04-tool-use/notebook-summary.md",
            chunk_id="tool-use-chunk-1",
        )
    ]

    context = build_context_block(chunks)

    assert "引用来源: /tmp/04-tool-use/notebook-summary.md#tool-use-chunk-1" in context
