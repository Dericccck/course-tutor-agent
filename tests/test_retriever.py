# 测试检索逻辑是否符合当前项目预期：
# 1. 普通问答应命中 Tool Use 相关资料
# 2. 总结问题应命中对应章节资料
# 3. 学习路线问题应至少返回 Agent 相关资料

from app.loader import load_documents
from app.retriever import retrieve_documents


# 课程资料根目录
ROOT_DIR = "/Users/a1-6/Desktop/AIAgent/code"


def test_retrieve_tool_use_question():
    # 测试普通问答是否能命中 Tool Use 相关资料
    documents = load_documents(ROOT_DIR)
    query = "tool use 是什么，和 agent 有什么关系？"

    results = retrieve_documents(query, documents, top_k=5)
    titles = [item.title for item in results]

    assert any("04 Tool Use" in title for title in titles)


def test_retrieve_summary_question():
    # 测试总结请求是否能命中 07 Planning Design 对应资料
    documents = load_documents(ROOT_DIR)
    query = "帮我总结 07-planning-design 这一节在讲什么"

    results = retrieve_documents(query, documents, top_k=5)
    sources = [item.source for item in results]

    assert any("07-planning-design" in source for source in sources)


def test_retrieve_study_plan_question():
    # 测试学习路线问题是否至少能命中 Agent 相关资料
    documents = load_documents(ROOT_DIR)
    query = "如果我只学 1-* 和 2-*，想做一个 AIAgent 项目，请按课程模块给我安排学习顺序。"

    results = retrieve_documents(query, documents, top_k=5)
    sources = [item.source for item in results]

    assert any("ai-agents-for-beginners" in source for source in sources)
