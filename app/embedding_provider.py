# embedding / 向量检索 接口层
from typing import Protocol # 定义向量存储接口的协议

from schemas import DocumentChunk

# EmbeddingProvider 定义了一个协议，规定了任何向量存储实现都必须提供 embed_texts 和 embed_query 两个方法。这样我们就可以在后续的代码中依赖这个协议，而不需要关心具体的实现细节。
class EmbeddingProvider(Protocol):
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """将一组文本转换为向量。"""
        ...
        
    def embed_query(self, query: str) -> list[float]:
        """将单条查询转换为向量。"""
        ...
        
# 同样地，EmbeddingProvider 也先放一个占位实现，等后续完善了再替换掉。
class NotImplementedEmbeddingProvider:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError("Embedding provider is not implemented yet.")
    
    def embed_query(self, query: str) -> list[float]:
        raise NotImplementedError("Embedding provider is not implemented yet.")

# 这个函数负责把一个 DocumentChunk 转换成适合做 embedding 的纯文本字符串。我们希望这个字符串同时包含标题、标签和正文内容，以便后续的向量化能够综合考虑这些信息。
def build_embedding_text(chunk: DocumentChunk) -> str:
    return (
        f"标题：{chunk.title}\n"
        f"标签：{', '.join(chunk.tags) if chunk.tags else '无'}\n"
        f"内容：{chunk.content}"
    )