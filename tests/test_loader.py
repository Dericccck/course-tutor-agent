# 先只测一件事：loader.py 能不能成功读到课程文件。
from pathlib import Path

from app.loader import build_tags, load_markdown_file
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
