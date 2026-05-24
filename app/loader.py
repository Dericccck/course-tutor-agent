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
def iter_course_files(root_dir: str, include_dirs: list[str] | None = None,) -> list[Path]:
    root = Path(root_dir)
    files: list[Path] = []

    # rglob("*.md") 是 "recursive glob" 的缩写。
    # 它会深入到 root 目录下的每一层子文件夹，寻找所有以 .md 结尾的文件。
    for path in root.rglob("*.md"):
        if not path.is_file():
            continue
        if is_excluded_path(path):
            continue
        if include_dirs and not is_included_course_path(path, include_dirs):
            continue
        if not is_allowed_course_file(path):
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
def load_documents(root_dir: str, include_dirs: list[str] | None = None,) -> list[Document]:
    documents: list[Document] = []

    for path in iter_course_files(root_dir, include_dirs):
        try:
            documents.append(load_markdown_file(path))
        except Exception as exc:
            print(f"Skip {path}: {exc}")

    return documents

def chunk_document(document: Document, max_chars: int = 500) -> list[DocumentChunk]:
    # 1. 第一级切分：按 Markdown 标题章节切分
    sections = split_markdown_sections(document.content)
    
    # 边缘情况：如果整篇文章没有一个 '#' 标题，导致切分不出章节
    if not sections:
        content = document.content.strip()
        if not content:
            return [] # 空白文档，直接返回空列表
        # 没有章节但有内容，整体作为一整块返回
        return [
            DocumentChunk(
                source=document.source,
                title=document.title,
                chunk_id=f"{document.title}-chunk-1",
                content=content,
                tags=document.tags,
            )
        ]
        
    chunks: list[DocumentChunk] = [] 
    current_parts: list[str] = []    # 用于收集并合并“小章节”的缓存列表
    current_length = 0               # 当前合并小章节的累计字符长度
    chunk_index = 1                  
    
    for section in sections:  
        section = section.strip()
        if not section:
            continue
        
        section_length = len(section)
        
        # --- 情况 A：单体章节字数直接爆表 (超限) ---
        if section_length > max_chars:
            # A-1: 先把之前积攒在 current_parts 里的那些“小章节”打包结算掉
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
                chunk_index += 1
                current_parts = [] # 清空，为后续腾地方
                current_length = 0
            
            # A-2: 开启第二级切分：将这个巨无霸章节打碎成段落
            paragraphs = [part.strip() for part in section.split("\n\n") if part.strip()]
            temp_parts: list[str] = []
            temp_length = 0
            
            for paragraph in paragraphs:
                paragraph_length = len(paragraph)
                
                # 如果段落积攒起来也超限了，就打包段落块
                if temp_parts and temp_length + paragraph_length + 2 > max_chars:
                    chunk_text = "\n\n".join(temp_parts).strip()
                    chunks.append(
                        DocumentChunk(
                            source=document.source,
                            title=document.title,
                            chunk_id=f"{document.title}-chunk-{chunk_index}",
                            content=chunk_text,
                            tags=document.tags,
                        )
                    )
                    chunk_index += 1
                    temp_parts = [paragraph]
                    temp_length = paragraph_length
                else:
                    temp_parts.append(paragraph)
                    # 修复点：确保只有在积攒了 2 个及以上段落时，才计入中间的 "\n\n" 长度
                    temp_length += paragraph_length + (2 if len(temp_parts) > 1 else 0)
                    
            # 别忘了打包这个巨无霸章节最后残留的段落
            if temp_parts:
                chunk_text = "\n\n".join(temp_parts).strip()
                chunks.append(
                    DocumentChunk(
                        source=document.source,
                        title=document.title,
                        chunk_id=f"{document.title}-chunk-{chunk_index}",
                        content=chunk_text,
                        tags=document.tags,
                    )
                )
                chunk_index += 1
                
            continue # 处理完巨无霸章节，直接跳过后面的普通章节聚合逻辑，进入下一次循环
        
        # --- 情况 B：正常的普通小章节 (未超限)，贪婪聚合它们 ---
        # 修复点：将原来的 paragraph_length 改为了符合当前作用域的 section_length
        if current_parts and current_length + section_length + 2 > max_chars:
            # 已经装不下了，打包当前累计的所有小章节
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
            chunk_index += 1          
            current_parts = [section] # 修复点：把新章节作为新块的排头兵
            current_length = section_length 
        else:
            current_parts.append(section)
            # 修复点：同样修复了长度“抢跑”计算
            current_length += section_length + (2 if len(current_parts) > 1 else 0)
        
    # 3. 终局收尾：把最后留在缓存里的普通小章节打包带走
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

def load_document_chunks(root_dir: str, include_dirs: list[str] | None = None,) -> list[DocumentChunk]:
    documents = load_documents(root_dir, include_dirs)
    chunks: list[DocumentChunk] = []
    
    for document in documents:
        chunks.extend(chunk_document(document))
    
    return chunks


# 整篇内容 -> markdown section -> chunk
def split_markdown_sections(content: str) -> list[str]:
    # 1. 将整篇 Markdown 内容按行切分
    lines = content.splitlines()
    sections: list[str] = []      # 存放最终切分出来的所有章节文本
    current_lines: list[str] = []  # 缓存当前正在收集的章节行

    for line in lines:
        stripped = line.strip()
        
        # 2. 核心判断：如果当前行是标题（以 # 开头），且当前缓存里已经有内容了
        if stripped.startswith("#") and current_lines:
            # 说明遇到了下一个新章节的起点，先把老章节打包
            section_text = "\n".join(current_lines).strip()
            if section_text:
                sections.append(section_text)
            
            # 另起炉灶：将当前的标题行作为新章节的第一行
            current_lines = [line]
        else:
            # 3. 如果不是标题，或者是文档开头的第一个标题（此时 current_lines 为空）
            # 直接将当前行追加到缓存中
            current_lines.append(line)

    # 4. 收尾工作：循环结束后，别忘了把最后留在缓存里的章节也打包带走
    if current_lines:
        section_text = "\n".join(current_lines).strip()
        if section_text:
            sections.append(section_text)

    return sections

# 过滤扫描到的文件路径，确保只处理那些包含在 course_include_dirs 列表中的目录下的文件。这有助于我们聚焦在课程内容相关的文档上，而不被项目实战等其他类型的文档干扰。 （扫描的文件目录白名单）
def is_included_course_path(path: Path, include_dirs: list[str]) -> bool:
    path_parts = set(path.parts)
    return any(include_dir in path_parts for include_dir in include_dirs)

ALLOWED_MARKDOWN_FILENAMES = {
    "notebook-summary.md",
    "knowledge-progression-map.md",
    "technology-depth-in-1-and-2-folders.md",
}

def is_allowed_course_file(path: Path) -> bool:
    return path.name in ALLOWED_MARKDOWN_FILENAMES
