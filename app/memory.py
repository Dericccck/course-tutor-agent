# 这个文件专门负责： 读本地 JSON   写本地 JSON    返回一个统一结构
import json
from pathlib import Path

MEMORY_FILE = Path(__file__).resolve().parents[1] / "data" / "user_memory.json"

def load_user_memory() -> dict:
    if not MEMORY_FILE.exists():
        return build_default_memory()
    
    with MEMORY_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)

def save_user_memory(memory: dict) -> None:
    # 健壮性设计：确保 data 文件夹存在。
    # parents=True 表示如果上级目录也不存在则一并创建
    # exist_ok=True 表示如果文件夹已存在，则静默跳过，不报错
    MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)

    with MEMORY_FILE.open("w", encoding="utf-8") as f:
        # json.dump 核心参数：
        # - ensure_ascii=False：允许直接写入中文等非 ASCII 字符，避免变成 \u4e00 这种乱码
        # - indent=2：设置缩进为 2 个空格，让生成的 JSON 文件美观可读，也极其方便 Git 进行版本内容对比
        json.dump(memory, f, ensure_ascii=False, indent=2)

def build_default_memory() -> dict:
    return {
        "learning_goal": "",
        "preferred_scope": "",
        "completed_topics": [],
        "recent_focus": "",
        "recent_focus_history": [],
    }

def update_recent_focus(memory: dict, focus_text: str) -> None:
    """
    更新用户的最近关注点
    """
    text = focus_text.strip()
    if not text:
        return
    
    memory["recent_focus"] = text # 最近在学什么

    history = memory.setdefault("recent_focus_history", [])
    history = [item for item in history if item != text] # 去重
    history.append(text) # 插入到最前面
    memory["recent_focus_history"] = history[-3:] # 保留最近 3 条
