import os
from dataclasses import dataclass

# pyrefly: ignore [missing-import]
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    llm_provider: str
    model_name: str
    api_key: str
    base_url: str | None
    course_source_root: str
    retrieval_top_k: int
    retrieval_mode: str
    course_include_dirs: list[str] # 通过环境变量配置一些特定的子目录，来让检索更聚焦一些。比如只检索“课程内容”相关的文档，而不检索“项目实战”相关的文档。(扫描的文件目录白名单)
    embedding_provider: str # 如果有多个 embedding 提供商（比如 OpenAI 的 embedding API、Azure 的 embedding API、或者本地的 embedding 模型等），可以通过这个字段来区分不同的 embedding 提供商，方便我们在代码里根据配置来选择使用哪个 embedding 提供商的接口来生成向量表示。
    embedding_model_name: str # 这个字段可以用来指定生成向量表示时使用的具体模型名称。
    embedding_cache_dir: str | None # 这个字段可以用来指定一个本地目录，用于缓存生成的向量表示。这样在后续的使用中，如果同样的文本需要生成向量表示时，我们就可以直接从缓存中读取，而不需要重复调用 embedding API 来生成，节省时间和计算资源。
    reranker_provider: str
    reranker_model_name: str
    reranker_cache_dir: str | None


def get_settings() -> Settings:
    llm_provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    model_name = os.getenv("MODEL_NAME", "gpt-4.1-mini").strip()
    retrieval_mode=os.getenv("RETRIEVAL_MODE", "chunk").strip().lower()
    course_source_root = os.getenv(
        "COURSE_SOURCE_ROOT",
        "/Users/a1-6/Desktop/AIAgent/code",
    ).strip()
    raw_include_dirs = os.getenv("COURSE_INCLUDE_DIRS", "1-3-SafeandReliableAIviaGuardrails,2-1-RetrievalAugmentedGeneration,2-2-BuildingAndEvaluatingAdvancedRAGApplications,2-3-ai-agents-for-beginners").strip()
    course_include_dirs = [item.strip() for item in raw_include_dirs.split(",") if item.strip()]
    retrieval_top_k = int(os.getenv("RETRIEVAL_TOP_K", "5"))
    embedding_provider = os.getenv("EMBEDDING_PROVIDER", "hash").strip().lower()
    embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3").strip()
    embedding_cache_dir = os.getenv("EMBEDDING_CACHE_DIR", "").strip() or None
    reranker_provider = os.getenv("RERANKER_PROVIDER", "none").strip().lower()
    reranker_model_name = os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-base").strip()
    reranker_cache_dir = os.getenv("RERANKER_CACHE_DIR", "").strip() or embedding_cache_dir

    if llm_provider == "github":
        api_key = os.getenv("GITHUB_TOKEN", "").strip()
        base_url = os.getenv(
            "GITHUB_MODELS_BASE_URL",
            "https://models.inference.ai.azure.com/",
        ).strip()
    else:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None

    return Settings(
        llm_provider=llm_provider,
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,
        course_source_root=course_source_root,
        retrieval_top_k=retrieval_top_k,
        retrieval_mode=retrieval_mode,
        course_include_dirs=course_include_dirs,
        embedding_provider=embedding_provider,
        embedding_model_name=embedding_model_name,
        embedding_cache_dir=embedding_cache_dir,
        reranker_provider=reranker_provider,
        reranker_model_name=reranker_model_name,
        reranker_cache_dir=reranker_cache_dir,
    )


def validate_settings(settings: Settings) -> None:
    if settings.llm_provider not in {"openai", "github"}:
        raise ValueError("LLM_PROVIDER must be either 'openai' or 'github'.")

    if not settings.api_key:
        if settings.llm_provider == "github":
            raise ValueError("GITHUB_TOKEN is required when LLM_PROVIDER=github.")
        raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai.")
    
    if settings.retrieval_mode not in {"document", "chunk", "vector", "hybrid"}: # 目前支持三种检索模式：最原始的文档级检索（document），更细粒度的切块级检索（chunk），以及基于向量搜索的检索（vector）。用户可以根据自己的需求和数据规模来选择合适的检索模式。一般来说，chunk 模式在大多数情况下会有更好的性能和相关性，而 vector 模式则适用于需要处理非常大规模文本数据或者需要更复杂语义理解的场景。
        raise ValueError("RETRIEVAL_MODE must be 'document', 'chunk', 'vector', or 'hybrid'.")
    
    if settings.embedding_provider not in {"hash", "sentence-transformers"}: # 目前先支持两种 embedding 提供商，后续如果需要支持更多，可以继续扩展这个条件判断。
        raise ValueError("EMBEDDING_PROVIDER must be 'hash' or 'sentence-transformers'.")
    
    if settings.reranker_provider not in {"none", "sentence-transformers"}: # 目前先支持两种 reranker 提供商，后续如果需要支持更多，可以继续扩展这个条件判断。
        raise ValueError("RERANKER_PROVIDER must be 'none' or 'sentence-transformers'.")

