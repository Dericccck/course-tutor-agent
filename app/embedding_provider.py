# embedding / 向量检索 接口层
# 这个文件定义了 EmbeddingProvider 协议和一些相关的类和函数。EmbeddingProvider 协议规定了任何向量存储实现都必须提供 embed_texts 和 embed_query 两个方法。我们还提供了一个 NotImplementedEmbeddingProvider 作为占位实现，以及一个 build_embedding_text 函数来把 DocumentChunk 转换成适合做 embedding 的文本字符串。

from typing import Protocol # 定义向量存储接口的协议

from schemas import DocumentChunk

import hashlib

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

# 这个函数根据配置的 provider_name 来构建一个 EmbeddingProvider 实例。当前我们只实现了一个简单的 HashEmbeddingProvider，实际使用时应该替换成一个真正的 embedding provider，比如调用 OpenAI 的 embedding API。
def build_embedding_provider(provider_name: str = "hash", model_name: str = "BAAI/bge-m3", cache_folder: str | None = None,) -> EmbeddingProvider:
    # 目前我们只实现了一个简单的 HashEmbeddingProvider，实际使用时应该替换成一个真正的 embedding provider，比如调用 OpenAI 的 embedding API。 --- IGNORE ---
    if provider_name == "hash":
        return HashEmbeddingProvider()
    elif provider_name == "sentence-transformers":
        return SentenceTransformerEmbeddingProvider(model_name=model_name, cache_folder=cache_folder)

    return NotImplementedEmbeddingProvider()

# 下面是一个简单的 HashEmbeddingProvider 实现，它通过对输入文本进行哈希计算来生成一个固定维度的向量表示。这个实现只是为了测试和占位，实际使用时应该替换成一个真正的 embedding provider，比如调用 OpenAI 的 embedding API。
class HashEmbeddingProvider:
    def __init__(self, dim: int = 8):
        self.dim = dim
    # 这个简单的实现通过对输入文本进行哈希计算来生成一个固定维度的向量表示。虽然这种方法没有真正捕捉文本的语义信息，但它可以用来测试整体的数据流和接口设计。实际使用时应该替换成一个真正的 embedding provider，比如调用 OpenAI 的 embedding API。
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]
    # embed_query 方法直接调用 embed 方法，因为在这个简单的实现中，文本和查询的处理方式是一样的。实际使用时，如果查询和文本需要不同的处理逻辑，也可以在这里进行区分。
    def embed_query(self, query: str) -> list[float]:
        return self._embed(query)
    # 这个简单的实现通过对输入文本进行哈希计算来生成一个固定维度的向量表示。虽然这种方法没有真正捕捉文本的语义信息，但它可以用来测试整体的数据流和接口设计。实际使用时应该替换成一个真正的 embedding provider，比如调用 OpenAI 的 embedding API。
    def _embed(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        # 将哈希值转换成一个固定维度的浮点数列表，范围在 0.0 到 1.0 之间。这里我们简单地把每个字节除以 255.0 来归一化。实际使用时，embedding provider 会返回一个更高维度且具有语义信息的向量。
        values: list[float] = []
        for i in range(self.dim):
            byte_value = digest[i]
            values.append(float(byte_value) / 255.0)
            
        return values
    

# 下面是一个基于 SentenceTransformer 的 EmbeddingProvider 实现。这个实现会使用指定的预训练模型来生成文本的向量表示。实际使用时需要安装 sentence-transformers 库，并且根据配置指定合适的模型名称。
class SentenceTransformerEmbeddingProvider:
    def __init__(self, model_name: str, cache_folder: str | None = None):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, cache_folder=cache_folder) # 这里我们指定了 cache_folder 来控制模型文件的缓存位置，避免每次都重新下载模型。实际使用时，可以根据需要调整这个路径。 , local_files_only=False
    
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(texts, normalize_embeddings=True) # 生成文本的向量表示，并进行归一化处理，使得每个向量的长度为 1。实际使用时，这样可以方便后续的相似度计算。
        return [embedding.tolist() for embedding in embeddings]
    
    def embed_query(self, query: str) -> list[float]:
        embedding = self.model.encode([query], normalize_embeddings=True)[0] # 生成查询的向量表示，并进行归一化处理，使得每个向量的长度为 1。实际使用时，这样可以方便后续的相似度计算。
        return embedding.tolist()