# 先只测一件事：loader.py 能不能成功读到课程文件。
from pathlib import Path

from app.loader import (
    build_tags,
    chunk_document,
    load_markdown_file,
    split_markdown_sections,
)
from app.schemas import Document


def test_build_tags_returns_list() -> None:
    path = Path("/tmp/2-3-ai-agents-for-beginners/rag/guardrails/lesson.md")
    assert build_tags(path) == ["agent", "guardrails", "rag"]


def test_document_accepts_none_tags() -> None:
    doc = Document(
        source="test.md",
        title="Test",
        content="content",
        doc_type="md",
        tags=None,
    )

    assert doc.tags == []


def test_load_markdown_file_populates_tags(tmp_path: Path) -> None:
    md_file = tmp_path / "2-3-ai-agents-for-beginners" / "rag" / "lesson.md"
    md_file.parent.mkdir(parents=True)
    md_file.write_text("# Lesson\nhello", encoding="utf-8")

    doc = load_markdown_file(md_file)

    assert doc.title == "Lesson"
    assert doc.tags == ["agent", "rag"]


def test_split_markdown_sections_splits_on_headings() -> None:
    # Markdown 内容应优先按标题边界切成多个 section
    content = """# 总标题

开头说明

## 小节一
内容 A

## 小节二
内容 B
"""

    sections = split_markdown_sections(content)

    assert len(sections) == 3
    assert sections[0].startswith("# 总标题")
    assert sections[1].startswith("## 小节一")
    assert sections[2].startswith("## 小节二")


def test_chunk_document_respects_markdown_sections() -> None:
    # 切块时应优先保留 section 边界，而不是随意打断标题块
    document = Document(
        source="/tmp/test.md",
        title="测试标题",
        content="""# 总标题

开头说明

## 第一部分
这里是第一部分的内容。

## 第二部分
这里是第二部分的内容。
""",
        doc_type="md",
        tags=["agent"],
    )

    chunks = chunk_document(document, max_chars=80)

    assert chunks
    assert chunks[0].content.startswith("# 总标题")
    assert any("## 第一部分" in chunk.content for chunk in chunks)
    assert any("## 第二部分" in chunk.content for chunk in chunks)


def test_chunk_document_falls_back_to_paragraph_split_for_long_section() -> None:
    # 单个 section 过长时，应退回按段落切块，而不是只生成一个超长 chunk
    long_paragraph_1 = "A" * 120
    long_paragraph_2 = "B" * 120
    document = Document(
        source="/tmp/test.md",
        title="超长章节测试",
        content=f"""# 标题

## 超长章节
{long_paragraph_1}

{long_paragraph_2}
""",
        doc_type="md",
        tags=["agent"],
    )

    chunks = chunk_document(document, max_chars=100)

    assert len(chunks) >= 2
    assert all(chunk.content.strip() for chunk in chunks)
