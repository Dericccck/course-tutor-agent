from typing import Protocol

from schemas import RetrievedChunk

class Reranker(Protocol):
    def reranker(self, query: str, chunks: list[RetrievedChunk], top_k: int = 5) -> list[RetrievedChunk]:
        """对检索结果进行重新排序。"""
        ...

class NotImplementedReranker:
    def reranker(self, query: str, chunks: list[RetrievedChunk], top_k: int = 5) -> list[RetrievedChunk]:
        """这个简单的实现直接返回原始的检索结果，没有进行任何重新排序。实际使用时应该替换成一个真正的 reranker，比如基于 LLM 的 reranker。"""
        raise NotImplementedError("Reranker is not implemented yet.")
    
class FakeReranker:
    def __init__(self, results: list[RetrievedChunk] | None = None):
        self.results = results or []
        self.calls: list[tuple[str, list[RetrievedChunk], int]] = [] # 记录每次调用的参数，方便测试时进行断言验证

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int = 5) -> list[RetrievedChunk]:
        """这个简单的实现直接返回预设的结果列表，或者如果没有预设结果，则返回原始的检索结果。
        它还会记录每次调用的参数，方便测试时进行断言验证。实际使用时应该替换成一个真正的 reranker，比如基于 LLM 的 reranker。"""
        self.calls.append((query, chunks, top_k))
        if self.results:
            return self.results[:top_k]
        return chunks[:top_k]

    
def build_reranker_text(chunk: RetrievedChunk) -> str:
    """这个函数负责把一个 RetrievedChunk 转换成适合做 reranking 的纯文本字符串。我们希望这个字符串同时包含标题、标签和正文内容，以便后续的 reranking 能够综合考虑这些信息。"""
    tags = ", ".join(chunk.tags) if chunk.tags else "无"
    return (
        f"标题：{chunk.title}\n"
        f"标签：{tags}\n"
        f"内容：{chunk.snippet}"
    )

class SentenceTransformersReranker:
    def __init__(self, model_name: str, cache_folder: str | None = None):
        from sentence_transformers import CrossEncoder # 这里我们使用了 SentenceTransformer 库中的 CrossEncoder 模型来实现 reranking。这个模型可以同时考虑查询和文本之间的交互信息，通常能够提供更准确的相关性评分。实际使用时应该替换成一个真正的 reranker，比如基于 LLM 的 reranker。
        self.model = CrossEncoder(model_name, cache_folder=cache_folder, local_files_only=True)

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int = 5) -> list[RetrievedChunk]:
        """这个实现通过调用 CrossEncoder 模型来对检索结果进行重新排序。
        首先，我们把每个 RetrievedChunk 转换成一个适合做 reranking 的文本字符串，然后把查询和这些文本字符串组成一对输入，传给模型进行评分。
        最后，我们根据模型返回的分数对检索结果进行排序，并返回 top_k 条结果。实际使用时应该替换成一个真正的 reranker，比如基于 LLM 的 reranker。"""
        if not chunks:
            return []
        
        pairs = [(query, build_reranker_text(chunk)) for chunk in chunks]
        scores = self.model.predict(pairs)

        rerankerd = [
            RetrievedChunk(
                source=chunk.source,
                chunk_id=chunk.chunk_id,
                title=chunk.title,
                tags=chunk.tags,
                snippet=chunk.snippet,
                score=score,
            )
            for chunk, score in zip(chunks, scores)
        ]
        rerankerd.sort(key=lambda item: item.score, reverse=True)
        return rerankerd[:top_k]
    
def build_reranker(provider_name: str = "none", model_name: str = "BAAI/bge-reranker-base", cache_folder: str | None = None,) -> Reranker | None:
    """这个函数根据配置的 provider_name 来构建一个 Reranker 实例。
    当前我们只实现了一个简单的 SentenceTransformersReranker，实际使用时应该替换成一个真正的 reranker，比如基于 LLM 的 reranker。"""
    if provider_name == "none":
        return None
    if provider_name == "sentence-transformers":
        return SentenceTransformersReranker(model_name=model_name, cache_folder=cache_folder)
    return NotImplementedReranker()