# 做最简单的本地检索。先不用向量库，只做关键词匹配和打分。
from pathlib import Path
import re
from schemas import Document, RetrievedChunk, DocumentChunk

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
    source_text = document.source.lower()


    score = 0.0

    for token in query_tokens:
        if token in title_text:
            score += 3.0
        if token in tag_text:
            score += 2.0
        if token in content_text:
            score += 1.0
        if token in source_text:
            score += 3.0

    file_name = Path(document.source).name.lower()
    if file_name == "notebook-summary.md":
        score += 2.0

    if is_study_plan_query(query):
        if "notebook-summary.md" in source_text:
            score += 1.5

        if "lesson" in title_text or "intro" in title_text:
            score += 1.5

        if "/1-" in source_text or "/2-" in source_text:
            score += 1.5

        if "这一节在做什么" in content_text or "关键收获" in content_text:
            score += 1.0

    if is_agent_project_query(query):
        if "ai-agents-for-beginners" in source_text:
            score += 2.0

        if "agent" in tag_text:
            score += 1.5

        if "agent" in title_text:
            score += 1.5
    
    query_phrases = extract_query_phrases(query)
    for phrase in query_phrases:
        if phrase and phrase in title_text:
            score += 6.0
    
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

def score_chunk(query: str, chunk: DocumentChunk) -> float:
    query_tokens = tokenize(query)
    query_phrases = extract_query_phrases(query)
    
    if not query_tokens and not query_phrases:
        return 0.0
    
    title_text = chunk.title.lower()
    content_text = chunk.content.lower()
    tag_text = " ".join(chunk.tags).lower()
    source_text = chunk.source.lower()
    
    score = 0.0
    
    for phrase in query_phrases:
        if phrase and phrase in title_text:
            score += 6.0
        if phrase and phrase in content_text:
            score += 3.0
            
    for token in query_tokens:
        if token in title_text:
            score += 3.0
        if token in tag_text:
            score += 2.0
        if token in content_text:
            score += 1.5
        if token in source_text:
            score += 3.0
    
    if "notebook-summary.md" in source_text:
        score += 1.5
        
    if is_study_plan_query(query):
        if "lesson" in title_text or "intro" in title_text:
            score += 1.0

        if "/1-" in source_text or "/2-" in source_text:
            score += 1.0

        if "这一节在做什么" in content_text or "关键收获" in content_text:
            score += 1.0
            
    if is_agent_project_query(query):
        if "ai-agents-for-beginners" in source_text:
            score += 2.0
        if "agent" in tag_text:
            score += 1.5
        if "agent" in title_text:
            score += 1.5
            
    return score

def build_chunk_snippet(chunk: DocumentChunk, query: str, max_length: int = 220) -> str:
    content = chunk.content.strip()
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

def retrieve_chunks(query: str, chunks: list[DocumentChunk], top_k: int = 5) -> list[RetrievedChunk]:
    scored_results: list[RetrievedChunk] = []
    
    for chunk in chunks:
        score = score_chunk(query, chunk)
        if score <= 0:
            continue
        
        scored_results.append(
            RetrievedChunk(
                source=chunk.source,
                title=chunk.title,
                snippet=build_chunk_snippet(chunk, query),
                score=score,
                tags=chunk.tags,
            )
        )
    
    scored_results.sort(key=lambda item: item.score, reverse=True)
    final_results: list[RetrievedChunk] = []
    source_counts: dict[str, int] = {}
    for item in scored_results:
        count = source_counts.get(item.source, 0)
        if count >= 2:  # 每个文档最多保留2个切分结果，避免过度集中
            continue
        
        final_results.append(item)
        source_counts[item.source] = count + 1
        
        if len(final_results) >= top_k:
            break
        
    return final_results
    

def is_study_plan_query(query: str) -> bool:
    lowered = query.lower()

    keywords = [
        "学习顺序",
        "学习路线",
        "学习计划",
        "怎么学",
        "从哪里开始",
        "先学什么",
        "roadmap",
        "plan",
    ]

    return any(keyword in lowered for keyword in keywords)

def is_agent_project_query(query: str) -> bool:
    lowered = query.lower()

    keywords = [
        "agent",
        "aiagent",
        "aiaagent",
        "智能体",
        "项目",
    ]

    return any(keyword in lowered for keyword in keywords)

# 短语提取函数：从用户问题中提取出一些可能有用的关键词或短语，供后续检索时加权使用。
def extract_query_phrases(query: str) -> list[str]:
    lowered = query.lower().strip()

    phrases: list[str] = []

    if len(lowered) >= 2:
        phrases.append(lowered)

    return phrases
            