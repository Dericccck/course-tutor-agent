# 放数据结构。至少要有：Document、RetrievedChunk、AgentAnswer
# pyrefly: ignore [missing-import]
from pydantic import BaseModel, Field, field_validator

# 一份原始课程资料
class Document(BaseModel):
    source: str = Field(..., description="Original file path of the document") # 文档的原始文件路径 “...”：这个字段没有默认值，创建对象时必须传
    title: str = Field(..., description="Human-readable document title") # 人类可读的文档标题
    content: str = Field(..., description="Full extracted text content") # 全文摘录
    doc_type: str = Field(..., description="Document type, e.g. md or ipynb") # 文档类型，例如md或ipynb
    tags: list[str] = Field(default_factory=list, description="Optional topic tags") # 可选主题标签 default_factory=list 表示“如果用户没传值，就默认创建一个新的空列表”

# 一次检索命中的结果
class RetrievedChunk(BaseModel):
    source: str = Field(..., description="Document source path") # 文档源路径
    title: str = Field(..., description="Document title") # 文件标题
    snippet: str = Field(..., description="Mateched text snippet") # 匹配的文本片段
    score: float = Field(..., description="Retrieval score") # 信息检索得分
    tags: list[str] = Field(default_factory=list, description="Optional topic tags") # 可选主题标签

# 最终返回给用户的结构化答案
class AgentAnswer(BaseModel):
    answer: str = Field(..., description="Direct answer to the user question") # 直接回答用户的问题
    suggestions: list[str] = Field(default_factory=list, description="Follow-up study suggestions") # 后续研究建议
    sources: list[str] = Field(default_factory=list, description="Referenced document sources") # 引用的文档源
