# 把 main.py 里的“向量索引构建/缓存恢复”抽成一个可复用服务
from schemas import DocumentChunk
from embedding_provider import build_embedding_provider
from vector_index_cache import (
    build_vector_index_payload,
    is_vector_index_cache_valid,
    load_vector_index_cache,
    save_vector_index_cache,
)
from vector_store import InMemoryVectorStore

def build_vector_store_with_cache(settings, chunks: list[DocumentChunk]) -> InMemoryVectorStore:
    """构建一个带缓存机制的向量存储服务。这个函数会先尝试从缓存中加载向量索引，如果缓存有效就直接使用，否则就重新计算 embeddings 并更新缓存。"""
    """
    构建一个可用的向量索引。

    优先级：
    1. 如果配置了 vector_index_cache_path，并且缓存文件存在且有效，就直接从缓存加载向量索引。
    2. 否则，就通过 embedding_provider 计算所有 chunks 的向量表示，并把结果写入缓存文件（如果配置了 vector_index_cache_path）。
    """
    embedding_provider = build_embedding_provider(
        settings.embedding_provider,
        model_name=settings.embedding_model_name,
        cache_folder=settings.embedding_cache_dir,
    )
    vector_store = InMemoryVectorStore(embedding_provider)
    cache_loaded = False

    # 尝试加载向量索引缓存，如果缓存有效就直接用缓存来恢复向量索引，跳过 embedding 计算的过程，显著提升启动速度；如果缓存无效（比如配置发生了变化，或者 chunks 内容发生了变化），就正常走向量索引构建流程，构建完成后再把新的向量索引缓存写回本地文件。
    if settings.vector_index_cache_path:
        cache_payload = load_vector_index_cache(settings.vector_index_cache_path)

        if cache_payload and is_vector_index_cache_valid(cache_payload, settings, chunks): # 命中缓存 - 只有当缓存存在且有效时，才使用缓存来恢复向量索引。这样我们就能确保在配置或者数据发生变化时，能够正确地重建向量索引，而不是误用过期的缓存数据。
            print("加载到有效的向量索引缓存，正在使用缓存来恢复向量索引...")
            cached_chunks = [DocumentChunk(**item) for item in cache_payload["chunks"]]
            cached_embeddings = cache_payload["embeddings"]

            vector_store.load_index(cached_chunks, cached_embeddings) # 不再重新算 embedding，而是直接用缓存中的 chunks 和 embeddings 来恢复向量索引，这样能显著提升启动速度，尤其是当 chunks 数量较大时。
            cache_loaded = True
            print("向量索引已从缓存中恢复。")
    
    if not cache_loaded:
        print("没有找到有效的向量索引缓存，正在构建向量索引...")
        vector_store.index_chunks(chunks) # 自动重建
        print("向量索引构建完成。")
        if settings.vector_index_cache_path:
            cache_payload = build_vector_index_payload(settings, chunks, vector_store.embeddings)
            save_vector_index_cache(settings.vector_index_cache_path, cache_payload) # 自动覆盖缓存文件
            print("新的向量索引缓存已保存到本地文件。")

    return vector_store

