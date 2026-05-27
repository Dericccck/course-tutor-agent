# 向量索引持久化
# 解决 3 件事： 1、避免每次启动都重新算 embeddings  2、让 vector / hybrid / reranker 调试更快   3、保持实现简单，可测试，可回退

# 这版先不做： FAISS / Chroma   增量更新单个chunk     多版本索引管理      mmap / 大规模优化
# 这版只做：  1、单文件索引缓存    2、命中就直接加载    3、失效就整包重建
import hashlib
import json
from pathlib import Path

from schemas import DocumentChunk

# 所有 chunk 按固定顺序拼起来。
# 这样只要：
    # 文本变了
    # chunk 切分变了
    # 标题变了
    # tags 变了
# 指纹就会变。
def build_chunk_fingerprint(chunks: list[DocumentChunk]) -> str:
    """构建一个课程切块列表的指纹，用于判断缓存是否失效。"""
    """基于当前chunks内容生成稳定指纹，用于判断缓存是否失效"""
    hasher = hashlib.sha256()

    for chunk in chunks:
        tags_text = ",".join(chunk.tags)
        payload = "\n".join(
            [
                chunk.source,
                chunk.title,
                chunk.chunk_id,
                chunk.content,
                tags_text,
            ]
        )
        hasher.update(payload.encode("utf-8")) # 把每个chunk的内容都加入哈希计算，这样只要chunks内容有任何变化，指纹就会改变，从而触发缓存重建。
        hasher.update(b"\n---\n")  # 分隔符，确保不同chunk之间的边界
    return hasher.hexdigest()

def load_vector_index_cache(cache_path: str) -> dict | None:
    """读取本地向量索引缓存，如果文件不存在则返回 None。"""
    path = Path(cache_path)
    if not path.exists():
        return None
    
    with path.open("r", encoding="utf-8") as f: # 读取本地向量索引缓存，如果文件不存在则返回 None - 这里我们使用了 json 格式来存储缓存数据，这样既方便人类阅读，也方便程序读取和写入。实际使用时，缓存数据中应该包含足够的信息来重建向量索引，比如 chunks 的指纹、每个 chunk 的向量表示等。
        return json.load(f)
    
def save_vector_index_cache(cache_path: str, payload: dict) -> None:
    """把向量索引缓存写到本地文件。"""
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)  # 确保目录存在

    with path.open("w", encoding="utf-8") as f: # 把向量索引缓存写到本地文件 - 这里我们使用了 json 格式来存储缓存数据，这样既方便人类阅读，也方便程序读取和写入。实际使用时，payload 中应该包含足够的信息来重建向量索引，比如 chunks 的指纹、每个 chunk 的向量表示等。
        json.dump(payload, f, ensure_ascii=False, indent=2)

def build_vector_index_payload(settings, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> dict:
    """构造可落盘的向量索引缓存结构。"""
    return {
        "version": 1,
        "embedding_provider": settings.retrieval.embedding_provider,
        "embedding_model_name": settings.retrieval.embedding_model_name,
        "course_source_root": settings.course_source_root,
        "course_include_dirs": settings.course_include_dirs,
        "chunks_fingerprint": build_chunk_fingerprint(chunks), # 通过 chunks 的指纹来判断缓存是否失效，如果指纹不匹配，就说明 chunks 内容发生了变化，需要重新计算 embeddings 并更新缓存。
        "chunks": [
            {
                "source": chunk.source,
                "title": chunk.title,
                "chunk_id": chunk.chunk_id,
                "content": chunk.content,
                "tags": chunk.tags,
            }
            for chunk in chunks
        ],
        "embeddings": embeddings, # 这里直接把所有 chunk 的向量表示都存储在缓存中，这样在加载缓存时就可以直接使用这些向量，而不需要重新计算。实际使用时，如果向量数据较大，也可以考虑只存储必要的信息，或者使用更高效的存储格式。
    }

# 以下都一致才复用：
    #  version
    #  embedding_provider
    #  embedding_model_name
    #  course_source_root
    #  course_include_dirs
    #  chunk_fingerprint
# 只要有一个不一致：
    #   直接重建索引
    #   然后覆盖写回缓存文件
def is_vector_index_cache_valid(payload: dict, settings, chunks: list[DocumentChunk]) -> bool:
    """判断加载的向量索引缓存是否仍然有效。"""
    """判断缓存是否仍然有效 - 通过比较缓存中的配置和当前的配置，以及 chunks 的指纹来判断。如果任何一个关键配置项发生了变化，或者 chunks 内容发生了变化，就说明缓存失效，需要重新计算 embeddings 并更新缓存。"""
    if payload.get("version") != 1:
        return False
    if payload.get("embedding_provider") != settings.retrieval.embedding_provider:
        return False
    if payload.get("embedding_model_name") != settings.retrieval.embedding_model_name:
        return False
    if payload.get("course_source_root") != settings.course_source_root:
        return False
    if payload.get("course_include_dirs") != settings.course_include_dirs:
        return False
    if payload.get("chunks_fingerprint") != build_chunk_fingerprint(chunks): # 通过比较缓存中的 chunks_fingerprint 和当前 chunks 的指纹来判断，如果不匹配，就说明 chunks 内容发生了变化，需要重新计算 embeddings 并更新缓存。
        return False
    if "chunks" not in payload or "embeddings" not in payload:
        return False
    
    return True