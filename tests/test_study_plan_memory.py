# 测试学习顺序建议与学习进度记忆的联动场景：
# 1. 已完成主题应进入“本次上下文中已完成的模块”
# 2. 未完成主题应进入“本次可优先推荐的未完成模块”
# 3. 学习路线 prompt 中应保留“避免重复推荐已完成主题”的约束

from prompts import build_study_plan_prompt
from schemas import RetrievedChunk


def make_chunk(title: str, source: str) -> RetrievedChunk:
    # 构造一个最小可用的检索结果对象，供学习路线 prompt 测试使用
    return RetrievedChunk(
        source=source,
        title=title,
        snippet=f"{title} 的相关片段",
        score=10.0,
        tags=["agent"],
    )


def test_study_plan_prompt_separates_completed_and_remaining_modules():
    # 学习路线 prompt 应把已完成模块和未完成模块明确区分开
    memory = {
        "learning_goal": "我想做一个 AIAgent 项目",
        "preferred_scope": "我只学习 1-* 和 2-* 的内容",
        "completed_topics": [
            "01 Intro To AI Agents 学习摘要",
            "07 Planning Design 学习摘要",
        ],
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
        make_chunk(
            "03 Agentic Design Patterns 学习摘要",
            "/tmp/03-agentic-design-patterns/notebook-summary.md",
        ),
    ]

    prompt = build_study_plan_prompt(
        "如果我想继续做一个 AIAgent 项目，接下来应该按课程模块怎么学？",
        chunks,
        memory=memory,
    )

    assert "本次可优先推荐的未完成模块" in prompt
    assert "本次上下文中已完成的模块" in prompt
    assert "- 01 Intro To AI Agents 学习摘要" in prompt
    assert "- 02 Explore Agentic Frameworks 学习摘要" in prompt
    assert "- 03 Agentic Design Patterns 学习摘要" in prompt
    assert "不要再把它们作为优先学习阶段重复推荐" in prompt
    assert "优先从这些模块中安排后续学习路线" in prompt


def test_study_plan_prompt_keeps_goal_scope_and_progress_together():
    # 学习路线 prompt 应同时携带目标、范围和已完成主题
    memory = {
        "learning_goal": "我想做一个 AIAgent 项目",
        "preferred_scope": "我只学习 1-* 和 2-* 的内容",
        "completed_topics": ["01 Intro To AI Agents 学习摘要"],
    }

    chunks = [
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

    assert "学习目标" in prompt
    assert "学习范围" in prompt
    assert "用户已完成的主题" in prompt
    assert "我想做一个 AIAgent 项目" in prompt
    assert "我只学习 1-* 和 2-* 的内容" in prompt
    assert "01 Intro To AI Agents 学习摘要" in prompt


def test_study_plan_prompt_requires_explaining_next_step_based_on_completed_topics():
    # 当存在已完成主题时，prompt 应明确要求模型说明“基于已完成内容，下一步建议是什么”
    memory = {
        "learning_goal": "我想继续做一个 AIAgent 项目",
        "preferred_scope": "我只学习 1-* 和 2-* 的内容",
        "completed_topics": [
            "01 Intro To AI Agents 学习摘要",
            "02 Explore Agentic Frameworks 学习摘要",
        ],
    }

    chunks = [
        make_chunk(
            "03 Agentic Design Patterns 学习摘要",
            "/tmp/03-agentic-design-patterns/notebook-summary.md",
        ),
        make_chunk(
            "05 Agentic RAG 学习摘要",
            "/tmp/05-agentic-rag/notebook-summary.md",
        ),
    ]

    prompt = build_study_plan_prompt(
        "如果我想继续做一个 AIAgent 项目，接下来应该按课程模块怎么学？",
        chunks,
        memory=memory,
    )

    assert "基于已完成内容，下一步建议是什么" in prompt
    assert "你已经完成了 A，因此建议继续学习 B、C" in prompt
