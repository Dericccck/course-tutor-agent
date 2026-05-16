# 测试任务分流逻辑是否正确：
# - 普通问答 -> qa
# - 课程总结 -> summary
# - 学习顺序建议 -> study_plan

from app.agent import detect_task_type


def test_detect_task_type_for_qa():
    # 普通概念问答应该被识别为 qa
    question = "tool use 是什么，和 agent 有什么关系？"
    assert detect_task_type(question) == "qa"


def test_detect_task_type_for_summary():
    # 带“总结”“这一节”等表达的问题应该被识别为 summary
    question = "帮我总结 07-planning-design 这一节在讲什么"
    assert detect_task_type(question) == "summary"


def test_detect_task_type_for_study_plan():
    # 带“学习顺序”“怎么安排”等表达的问题应该被识别为 study_plan
    question = "如果我只学 1-* 和 2-*，想做一个 AIAgent 项目，学习顺序应该怎么安排？"
    assert detect_task_type(question) == "study_plan"
