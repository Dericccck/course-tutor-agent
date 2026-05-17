# 扫描并读取课程资料。你的第一批重点就是读 notebook-summary.md、.md、部分 .ipynb。
from pathlib import Path
from schemas import Document, DocumentChunk

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

def chunk_document(document: Document, max_chars: int = 500) -> list[DocumentChunk]:
    # 1. 预处理：按双换行符切分出所有段落，并去除两边的空白字符，过滤掉空段落
    paragraphs = [part.strip() for part in document.content.split("\n\n") if part.strip()]
    
    # 2. 边界情况处理：如果文档是个空文件或没有有效段落
    if not paragraphs:
        # 直接把整篇空内容打包成一个块返回，防止后续逻辑报错
        return [
            DocumentChunk(
                source=document.source,
                title=document.title,
                chunk_id=f"{document.source}-chunk-1",
                content=document.content,
                tags=document.tags,
            )
        ]
        
    # 3. 初始化变量，准备进行贪婪聚合
    chunks: list[DocumentChunk] = [] # 存放最终分块结果的列表
    current_parts: list[str] = []    # 存放当前分块正在收集的段落
    current_length = 0               # 当前分块的累计字符长度
    chunk_index = 1                  # 分块计数器，用于生成唯一的 chunk_id
    
    # 4. 遍历每个段落进行聚合判断
    for paragraph in paragraphs:  # 💡 帮你把原代码的拼写错误 paragragh 修正为了 paragraph
        paragraph_length = len(paragraph)
        
        # 判断如果把当前段落加进去，是否会超过单块最大字符限制 (max_chars)
        # 加 2 是因为段落之间拼接时需要重新补上 "\n\n" 两个字符
        if current_parts and current_length + paragraph_length + 2 > max_chars:
            # 如果超限了，说明当前块已经“饱了”，先打包当前块
            chunk_text = "\n\n".join(current_parts).strip()
            chunks.append(
                DocumentChunk(
                    source=document.source,
                    title=document.title,
                    chunk_id=f"{document.title}-chunk-{chunk_index}",
                    content=chunk_text,
                    tags=document.tags,
                )
            )
            chunk_index += 1          # 计数器递增
            current_parts = [paragraph] # 另起炉灶，将当前段落作为新一块的开头
            current_length = paragraph_length # 重置新块的长度
        else:
            # 如果没超限，或者当前块还是空的，就把当前段落塞进当前块中
            current_parts.append(paragraph)
            current_length += paragraph_length + (2 if current_parts else 0)
        
    # 5. 收尾工作：循环结束后，如果当前块里还有残留的段落，千万别漏了，打包带走
    if current_parts:
        chunk_text = "\n\n".join(current_parts).strip()
        chunks.append(
            DocumentChunk(
                source=document.source,
                title=document.title,
                chunk_id=f"{document.title}-chunk-{chunk_index}",
                content=chunk_text,
                tags=document.tags,
            )
        )
    
    return chunks

def load_document_chunks(root_dir: str) -> list[DocumentChunk]:
    documents = load_documents(root_dir)
    chunks: list[DocumentChunk] = []
    
    for document in documents:
        chunks.extend(chunk_document(document))
    
    return chunks
