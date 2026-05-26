# embedding / 向量检索 接口层
from typing import Protocol # 定义向量存储接口的协议

from schemas import DocumentChunk, RetrievedChunk
from embedding_provider import EmbeddingProvider, build_embedding_text

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

# 目标不是马上接真实 embedding，而是先把未来真实向量检索的数据流打通
# 下面是一个 InMemoryVectorStore 的初步框架，虽然 search 方法还没实现，但先把整体结构搭起来了。这个类会在内存中保存 chunks 和它们的 embeddings，后续我们可以在 search 方法里实现基于余弦相似度的检索逻辑。
class InMemoryVectorStore:
    def __init__(self, embedding_provider: EmbeddingProvider):
        self.embedding_provider = embedding_provider
        self.chunks: list[DocumentChunk] = []
        self.embeddings: list[list[float]] = []
    
    # index_chunks 方法会把传入的 chunks 存起来，并通过 embedding_provider 生成它们的向量表示。这里我们用 build_embedding_text 把每个 chunk 转换成一个适合做 embedding 的文本字符串。
    def index_chunks(self, chunks: list[DocumentChunk]) -> None:
        self.chunks = list(chunks)
        texts = [build_embedding_text(chunk) for chunk in chunks]
        self.embeddings = self.embedding_provider.embed_texts(texts)

    # 后面 main.py 命中缓存时，就不用再：vector_store.index_chunks(chunks) 了，而是直接：vector_store.load_index(chunks, embeddings)，跳过 embedding 计算的过程。这样就能显著提升启动速度，尤其是当 chunks 数量较大时。
    def load_index(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        """直接从缓存恢复 chunks 和 embeddings。，跳过 embedding 计算的过程。这个方法可以用来从缓存中重建向量索引，避免每次启动都重新计算 embeddings。"""
        self.chunks = list(chunks)
        self.embeddings = list(embeddings)
    
    # search 方法的实现会比较复杂一些，因为需要计算查询的向量表示，然后跟所有 chunks 的向量进行相似度计算，最后返回最相关的 top_k 个结果。这里先放一个 NotImplementedError，等后续完善了再替换掉。
    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        if not self.chunks or not self.embeddings:
            return []
        # 首先把查询转换成向量表示
        query_embedding = self.embedding_provider.embed_query(query)
        
        scored_items: list[tuple[float, DocumentChunk]] = []
        # 然后计算查询向量和每个 chunk 向量的相似度得分，这里先用简单的点积，后续可以换成余弦相似度或者其他更合适的相似度计算方法。
        for chunk, embedding in zip(self.chunks, self.embeddings):
            score = dot_product(query_embedding, embedding) # 这里先用简单的点积，后续可以换成余弦相似度或者其他更合适的相似度计算方法
            scored_items.append((score, chunk))
        # 根据得分排序，取 top_k 个，并构建 RetrievedChunk 列表返回。这里我们直接用 chunk 的内容的前 220 字作为 snippet，后续可以改成更智能的摘要逻辑。
        scored_items.sort(key=lambda item: item[0], reverse=True)
        
        results: list[RetrievedChunk] = []
        # 根据得分排序，取 top_k 个，并构建 RetrievedChunk 列表返回。这里我们直接用 chunk 的内容的前 220 字作为 snippet，后续可以改成更智能的摘要逻辑。
        for score, chunk in scored_items[:top_k]:
            results.append(
                RetrievedChunk(
                    source=chunk.source,
                    title=chunk.title,
                    chunk_id=chunk.chunk_id,
                    snippet=chunk.content[:220].replace("\n", " ").strip(), # 直接用 chunk 内容的前 200 字作为 snippet，后续可以改成更智能的摘要
                    score=score,
                    tags=chunk.tags,
                )
            )
        return results
    
def dot_product(a : list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))