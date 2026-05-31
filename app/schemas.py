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
    # --- 构建前置数据清洗器 ---
    # @field_validator: Pydantic v2 的核心校验装饰器，指定对 "tags" 字段进行校验
    # mode="before": 关键参数！表示该逻辑在 Pydantic 进行标准类型检查和转换【之前】触发
    @field_validator("tags", mode="before")
    # Pydantic 的验证器方法通常需要声明为类方法
    @classmethod
    def normalize_tags(cls, value):
        """
        前置清洗函数：接收外界传入的原始脏数据（value），在它被强转为 list[str] 之前进行拦截。
        """
        if value is None:
            return []
        return value

# 一次检索命中的结果
class RetrievedChunk(BaseModel):
    source: str = Field(..., description="Document source path") # 文档源路径
    title: str = Field(..., description="Document title") # 文件标题
    chunk_id: str | None = Field(
        default=None,
        description="Optional chunk identifier for chunk-level retrieval results",
    )
    snippet: str = Field(..., description="Mateched text snippet") # 匹配的文本片段
    score: float = Field(..., description="Retrieval score") # 信息检索得分
    tags: list[str] = Field(default_factory=list, description="Optional topic tags") # 可选主题标签

# 最终返回给用户的结构化答案
class AgentAnswer(BaseModel):
    answer: str = Field(..., description="Direct answer to the user question") # 直接回答用户的问题
    suggestions: list[str] = Field(default_factory=list, description="Follow-up study suggestions") # 后续研究建议
    sources: list[str] = Field(default_factory=list, description="Referenced document sources") # 引用的文档源
    debug: dict = Field(default_factory=dict) # 调试字段

# 切分后的检索单元
class DocumentChunk(BaseModel):
    source: str = Field(..., description="Original document source path") # 文档源路径
    title: str = Field(..., description="Document title") # 文件标题
    chunk_id: str = Field(..., description="Unique chunk identifier") # 唯一标识符
    content: str = Field(..., description="Chunk text content") # 文本内容
    tags: list[str] = Field(default_factory=list, description="Optional topic tags") # 可选主题标签
