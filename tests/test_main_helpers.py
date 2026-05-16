# 测试 main.py 中当前可独立验证的 CLI 辅助输出：
# 1. 帮助文本应包含当前支持的核心命令
# 2. 示例问题应覆盖问答 / 总结 / 学习顺序建议三类能力

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
    assert "clear_goal" in output
    assert "clear_scope" in output
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
