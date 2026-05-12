# 扫描并读取课程资料。你的第一批重点就是读 notebook-summary.md、.md、部分 .ipynb。
from pathlib import Path
from schemas import Document

EXCLUDED_DIR_NAMES = {
    "venv",
    ".venv",
    "site-packages",
    "__pycache__",
    ".git",
    "node_modules",
    "Shared_data",
    "shared_data",
    ".ipynb_checkpoints",
}

def is_excluded_path(path: Path) -> bool:
    return any(part in EXCLUDED_DIR_NAMES for part in path.parts)

# 扫描目录下所有 .md 文件。
def iter_course_files(root_dir: str) -> list[Path]:
    root = Path(root_dir)
    files: list[Path] = []

    # rglob("*.md") 是 "recursive glob" 的缩写。
    # 它会深入到 root 目录下的每一层子文件夹，寻找所有以 .md 结尾的文件。
    for path in root.rglob("*.md"):
        if not path.is_file():
            continue
        if is_excluded_path(path):
            continue
        files.append(path)
    return sorted(files)

# 优先从 markdown 的一级标题里取标题；如果没有，就退回文件名。
def build_title(path: Path, content: str) -> str:
    lines = content.splitlines()

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
    # 返回路径对象中的 stem 属性（即不带后缀的文件名，如 "note.md" 的 stem 是 "note"）
    return path.stem

# 先做一个很粗糙的标签生成，后面可继续优化。
def build_tags(path: Path) -> list[str]:
    # 例如：路径 "AIAgent/RAG/paper.pdf" 会变成 ["aiagent", "rag", "paper.pdf"]
    parts = [part.lower() for part in path.parts]
    tags: list[str] = []

    if "ai-agents-for-beginners" in "-".join(parts):
        tags.append("agent")

    if "guardrails" in "-".join(parts):
        tags.append("guardrails")

    if "rag" in "-".join(parts):
        tags.append("rag")
    return tags

# 把单个 markdown 文件转成一个 Document。
def load_markdown_file(path: str | Path) -> Document: # 既可以接收一个普通的字符串（str），也可以接收一个 pathlib.Path 对象。
    # 1. 确保 path 是一个 Path 对象。
    # 无论传入的是字符串还是 Path 实例，都会统一转换为 pathlib.Path 以便进行后续操作。
    file_path = Path(path)
    # 2. 读取文件内容。
    # encoding="utf-8" 是最佳实践，能确保正确读取包含中文或特殊字符的 Markdown 笔记。
    content = file_path.read_text(encoding="utf-8")

    return Document(
        source=str(file_path),
        title=build_title(file_path, content),
        content=content,
        doc_type="md",
        tags=build_tags(file_path),
    )

# 批量加载所有文档，形成统一数据列表。
def load_documents(root_dir: str) -> list[Document]:
    documents: list[Document] = []

    for path in iter_course_files(root_dir):
        try:
            documents.append(load_markdown_file(path))
        except Exception as exc:
            print(f"Skip {path}: {exc}")

    return documents
