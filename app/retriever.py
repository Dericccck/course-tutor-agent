# 做最简单的本地检索。先不用向量库，只做关键词匹配和打分。
from pathlib import Path
import re
from schemas import Document, RetrievedChunk

# 把问题拆成词
def tokenize(text: str) -> list[str]:
    # 在构建 #RAG 系统时，使用了 bge-small-en-v1.5 模型！
    # 输出: ['在构建', 'rag', '系统时', '使用了', 'bge-small-en-v1.5', '模型']
    return re.findall(r"[a-zA-Z0-9_\-\u4e00-\u9fff]+", text.lower())

# 对每个文档打分：命中标题，加 3 分    命中标签，加 2 分    命中正文，加 1 分
def score_document(query: str, document: Document) -> float:
    query_tokens = tokenize(query)
    if not query_tokens:
        return 0.0
    
    title_text = document.title.lower()
    content_text = document.content.lower()
    tag_text = " ".join(document.tags).lower()

    score = 0.0

    for token in query_tokens:
        if token in title_text:
            score += 3.0
        if token in tag_text:
            score += 2.0
        if token in content_text:
            score += 1.0

    file_name = Path(document.source).name.lower()
    if file_name == "notebook-summary.md":
        score += 2.0
    
    return score

# 从文档里截一小段命中内容，后面给模型时会更有用，也便于调试。
def build_snippet(document: Document, query: str, max_length: int = 220) -> str:
    content = document.content.strip()
    if not content:
        return ""

    query_tokens = tokenize(query)

    for token in query_tokens:
        index = content.lower().find(token)
        if index != -1:
            start = max(0, index - 60)
            end = min(len(content), index + max_length)
            snippet = content[start:end].replace("\n", " ").strip()
            return snippet

    fallback = content[:max_length].replace("\n", " ").strip()
    return fallback

# 遍历全部文档，给每个文档算分，过滤掉 0 分文档，然后按分数排序，取前 top_k 个。
def retrieve_documents(query: str, documents: list[Document], top_k: int = 5,) -> list[RetrievedChunk]:
    scored_results: list[RetrievedChunk] = []

    for document in documents:
        score = score_document(query, document)
        if score <= 0:
            continue

        scored_results.append(
            RetrievedChunk(
                source=document.source,
                title=document.title,
                snippet=build_snippet(document, query),
                score=score,
                tags=document.tags,
            )
        )

    scored_results.sort(key=lambda item: item.score, reverse=True)
    return scored_results[:top_k]
            