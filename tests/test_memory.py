# 测试本地用户记忆的读写逻辑：
# 1. 默认结构函数返回统一 memory 结构
# 2. 首次读取时返回默认结构
# 3. 保存后能够正确读回
# 4. completed_topics 列表能被正确保留

from pathlib import Path

import memory


def test_build_default_memory_returns_expected_structure():
    # 默认 memory 结构应统一由 build_default_memory 提供
    result = memory.build_default_memory()

    assert result == {
        "learning_goal": "",
        "preferred_scope": "",
        "completed_topics": [],
        "recent_focus": "",
        "recent_focus_history": [],
    }


def test_load_user_memory_returns_default_when_file_missing(monkeypatch, tmp_path: Path):
    # 当 memory 文件不存在时，应返回默认结构
    test_memory_file = tmp_path / "user_memory.json"
    monkeypatch.setattr(memory, "MEMORY_FILE", test_memory_file)

    result = memory.load_user_memory()

    assert result == memory.build_default_memory()


def test_save_and_load_user_memory_round_trip(monkeypatch, tmp_path: Path):
    # 保存后的 memory 应能被完整读取回来
    test_memory_file = tmp_path / "user_memory.json"
    monkeypatch.setattr(memory, "MEMORY_FILE", test_memory_file)

    payload = {
        "learning_goal": "我想做一个 AIAgent 项目",
        "preferred_scope": "我只学习 1-* 和 2-* 的内容",
        "completed_topics": ["07 Planning Design 学习摘要"],
        "recent_focus": "最近在复习/总结：帮我总结 07-planning-design 这一节在讲什么",
        "recent_focus_history": ["07 Planning Design 总结"],
    }

    memory.save_user_memory(payload)
    loaded = memory.load_user_memory()

    assert loaded == payload


def test_save_user_memory_creates_parent_directory(monkeypatch, tmp_path: Path):
    # 保存 memory 时，如果父目录不存在，应自动创建
    test_memory_file = tmp_path / "nested" / "user_memory.json"
    monkeypatch.setattr(memory, "MEMORY_FILE", test_memory_file)

    payload = {
        "learning_goal": "测试目标",
        "preferred_scope": "测试范围",
        "completed_topics": [],
        "recent_focus": "",
        "recent_focus_history": [],
    }

    memory.save_user_memory(payload)

    assert test_memory_file.exists()


def test_update_recent_focus_overwrites_focus_text():
    payload = memory.build_default_memory()

    memory.update_recent_focus(payload, "最近在规划学习路线：RAG 学习顺序")

    assert payload["recent_focus"] == "最近在规划学习路线：RAG 学习顺序"
    assert payload["recent_focus_history"] == ["最近在规划学习路线：RAG 学习顺序"]


def test_update_recent_focus_keeps_recent_history_tail():
    payload = memory.build_default_memory()

    memory.update_recent_focus(payload, "Tool use 与 agent tool calling")
    memory.update_recent_focus(payload, "05 Agentic RAG 总结")
    memory.update_recent_focus(payload, "RAG 到 Agentic RAG 学习路线")
    memory.update_recent_focus(payload, "Planning agent 与任务拆解")

    assert payload["recent_focus"] == "Planning agent 与任务拆解"
    assert payload["recent_focus_history"] == [
        "05 Agentic RAG 总结",
        "RAG 到 Agentic RAG 学习路线",
        "Planning agent 与任务拆解",
    ]
