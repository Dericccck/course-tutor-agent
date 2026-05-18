# embedding / 向量检索 接口层
from typing import Protocol # 定义向量存储接口的协议

from schemas import DocumentChunk, RetrievedChunk

class EmbeddingProvider(Protocol):
    def embed_text(self, texts: list[str]) -> list[list[float]]:
        """将一组文本转换为向量。"""
        ...
        
    def embed_query(self, query:str) -> list[float]:
        """将单挑查询转换为向量。"""
        ...
        
class VectorStore(Protocol):
    def index_chunks(self, chunks: list[DocumentChunk]) -> None:
        """将 chunks 建立到向量索引中。"""
        ...
        
    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        """将查询返回最相关的检索结果"""
        ...

# 目前还没有真正的向量存储实现，所以先放一个占位的 NotImplementedVectorStore，等后续完善了再替换掉这个占位实现。  
class NotImplementedVectorStore:
    def index_chunks(self, chunks: list[DocumentChunk]) -> None:
        raise NotImplementedError("Vector store is not implemented yet.")
    
    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        raise NotImplementedError("Vector store search is not implemented yet.")
    
# 这个 FakeVectorStore 可以在测试时用来模拟向量检索的结果，方便我们先把整体流程搭起来，后续再替换成真正的向量存储实现。
class FakeVectorStore:
    def __init__(self, results: list[RetrievedChunk] | None = None):
        self.results = results or [] # 初始化时可以直接传预设检索结果
        self.indexed_chunks: list[DocumentChunk] = []
        
    def index_chunks(self, chunks: list[DocumentChunk]) -> None:
        self.indexed_chunks = list(chunks) # 把 chunks 存起来，不做任何索引逻辑。这样后面如果想测“索引过程有没有被调用”，也有抓手
        
    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        return self.results[:top_k]