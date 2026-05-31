# 测试 main.py 中当前可独立验证的 CLI 辅助输出：
# 1. 帮助文本应包含当前支持的核心命令
# 2. 示例问题应覆盖问答 / 总结 / 学习顺序建议三类能力
# 3. 学习进度摘要应正确展示空状态和有进度状态
# 4. 学习进度摘要应给出下一步推荐主题

import main


def test_print_help_contains_core_commands(capsys):
    # 帮助文本中应包含当前支持的关键命令
    main.print_help()
    captured = capsys.readouterr()
    output = captured.out

    assert "课程辅导 Agent 支持以下能力" in output
    assert "help" in output
    assert "set_goal:" in output
    assert "set_scope:" in output
    assert "mark_done:" in output
    assert "show_memory" in output
    assert "show_progress" in output
    assert "clear_goal" in output
    assert "clear_scope" in output
    assert "clear_done" in output
    assert "reset_memory" in output
    assert "unmark_done:" in output
    assert "exit / quit / q" in output


def test_print_example_questions_covers_three_core_scenarios(capsys):
    # 示例问题应覆盖普通问答、章节总结和学习顺序建议
    main.print_example_questions()
    captured = capsys.readouterr()
    output = captured.out

    assert "tool use 是什么，和 agent 有什么关系？" in output
    assert "帮我总结 07-planning-design 这一节在讲什么" in output
    assert "如果我只学 1-* 和 2-*，想做一个 AIAgent 项目，请按课程模块给我安排学习顺序。" in output


def test_should_show_retrieval_debug_defaults_to_false(monkeypatch):
    monkeypatch.delenv("SHOW_RETRIEVAL_DEBUG", raising=False)

    assert main.should_show_retrieval_debug() is False


def test_should_show_retrieval_debug_accepts_true_values(monkeypatch):
    for value in ["true", "1", "yes", "on"]:
        monkeypatch.setenv("SHOW_RETRIEVAL_DEBUG", value)
        assert main.should_show_retrieval_debug() is True


def test_print_help_uses_configured_texts(monkeypatch, capsys):
    monkeypatch.setattr(
        main,
        "load_cli_texts_config",
        lambda: {
            "help": {
                "capabilities_title": "自定义能力：",
                "capabilities": ["能力 A"],
                "commands_title": "自定义命令：",
                "commands": ["- 自定义命令"],
            }
        },
    )

    main.print_help()
    output = capsys.readouterr().out

    assert "自定义能力：" in output
    assert "能力 A" in output
    assert "自定义命令：" in output
    assert "- 自定义命令" in output


def test_print_progress_uses_configured_messages(monkeypatch, capsys):
    monkeypatch.setattr(
        main,
        "load_cli_texts_config",
        lambda: {
            "progress": {
                "title": "自定义进度：",
                "goal_unset": "未设目标",
                "scope_unset": "未设范围",
                "completed_topics_title": "完成列表:",
                "completed_topics_none": "暂无",
                "suggestion_when_started": "先打基础",
                "suggestion_when_in_progress": "继续推进",
                "next_topic_label": "下一步推荐",
                "next_topic_when_complete": "已经全部完成",
            }
        },
    )

    memory = {
        "learning_goal": "",
        "preferred_scope": "",
        "completed_topics": [],
        "recent_focus": "",
    }

    main.print_progress(memory)
    output = capsys.readouterr().out

    assert "自定义进度：" in output
    assert "学习目标: 未设目标" in output
    assert "学习范围: 未设范围" in output
    assert "完成列表: 暂无" in output
    assert "先打基础" in output
    assert "下一步推荐: 01 Intro To AI Agents 学习摘要" in output


def test_print_progress_shows_empty_state(capsys):
    # 当记忆为空时，学习进度摘要应显示未设置和空列表
    memory = {
        "learning_goal": "",
        "preferred_scope": "",
        "completed_topics": [],
        "recent_focus": "",
    }

    main.print_progress(memory)
    captured = capsys.readouterr()
    output = captured.out

    assert "当前学习进度摘要" in output
    assert "学习目标: 未设置" in output
    assert "学习范围: 未设置" in output
    assert "已完成主题数量: 0" in output
    assert "已完成主题列表: 无" in output
    assert "可以先从基础模块开始建立整体框架" in output
    assert "当前建议下一步: 01 Intro To AI Agents 学习摘要" in output


def test_print_progress_shows_completed_topics(capsys):
    # 当记忆中已有目标、范围和已完成主题时，应正确展示摘要信息
    memory = {
        "learning_goal": "我想做一个 AIAgent 项目",
        "preferred_scope": "我只学习 1-* 和 2-* 的内容",
        "completed_topics": [
            "01 Intro To AI Agents 学习摘要",
            "02 Explore Agentic Frameworks 学习摘要",
        ],
        "recent_focus": "Tool use 与 agent tool calling",
    }

    main.print_progress(memory)
    captured = capsys.readouterr()
    output = captured.out

    assert "学习目标: 我想做一个 AIAgent 项目" in output
    assert "学习范围: 我只学习 1-* 和 2-* 的内容" in output
    assert "已完成主题数量: 2" in output
    assert "01 Intro To AI Agents 学习摘要" in output
    assert "02 Explore Agentic Frameworks 学习摘要" in output
    assert "优先继续学习尚未完成的后续模块" in output
    assert "当前建议下一步: 03 Agentic Design Patterns 学习摘要" in output


def test_get_next_recommended_topic_returns_first_unfinished():
    # 当只完成了前两个主题时，下一步应推荐第三个主题
    memory = {
        "learning_goal": "",
        "preferred_scope": "",
        "completed_topics": [
            "01 Intro To AI Agents 学习摘要",
            "02 Explore Agentic Frameworks 学习摘要",
        ],
        "recent_focus": "",
    }

    result = main.get_next_recommended_topic(memory)

    assert result == "03 Agentic Design Patterns 学习摘要"


def test_get_next_recommended_topic_returns_none_when_all_finished():
    # 当主学习路线全部完成时，不应再返回新的推荐主题
    memory = {
        "learning_goal": "",
        "preferred_scope": "",
        "completed_topics": list(main.get_learning_sequence()),
        "recent_focus": "",
    }

    result = main.get_next_recommended_topic(memory)

    assert result is None


def test_get_learning_sequence_loads_default_order_from_config():
    # CLI 中的学习主线应直接复用 study_plan_order.json 的 default_title_order
    sequence = main.get_learning_sequence()

    assert sequence[0] == "01 Intro To AI Agents 学习摘要"
    assert sequence[1] == "02 Explore Agentic Frameworks 学习摘要"
    assert "05 Agentic RAG 学习摘要" in sequence


def test_get_learning_stage_uses_configured_stage_thresholds(monkeypatch):
    # 阶段判断应直接复用配置中的 learning_stages，而不是写死在代码里
    monkeypatch.setattr(
        main,
        "load_study_plan_order_config",
        lambda: {
            "default_title_order": [
                "模块 A",
                "模块 B",
                "模块 C",
            ],
            "learning_stages": [
                {"max_completed": 0, "label": "阶段 0"},
                {"max_completed": 1, "label": "阶段 1"},
                {"max_completed": 2, "label": "阶段 2"},
                {"max_completed": 999, "label": "阶段 3"},
            ],
        },
    )

    assert main.get_learning_stage({"completed_topics": []}) == "阶段 0"
    assert main.get_learning_stage({"completed_topics": ["模块 A"]}) == "阶段 1"
    assert main.get_learning_stage({"completed_topics": ["模块 A", "模块 B"]}) == "阶段 2"
    assert main.get_learning_stage({"completed_topics": ["模块 A", "模块 B", "模块 C"]}) == "阶段 3"


def test_build_recent_focus_from_question_formats_by_task_type():
    assert (
        main.build_recent_focus_from_question("tool use 是什么？", "qa")
        == "Tool use 与 agent tool calling"
    )
    assert (
        main.build_recent_focus_from_question("帮我总结 05-agentic-rag 这一节在讲什么", "summary")
        == "05 Agentic RAG 总结"
    )
    assert (
        main.build_recent_focus_from_question(
            "如果我想重点学 RAG，再过渡到 Agentic RAG，应该怎么安排学习顺序？",
            "study_plan",
        )
        == "RAG 到 Agentic RAG 学习路线"
    )
