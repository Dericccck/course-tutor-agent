import os
from dataclasses import dataclass

# pyrefly: ignore [missing-import]
from dotenv import load_dotenv

load_dotenv()

@dataclass
class RetrievalSettings: # 这个类用来专门存储检索相关的配置项，方便我们在代码里清晰地管理和使用这些配置。通过把检索相关的配置项集中在一个类里，我们可以更好地组织代码，并且在需要调整检索逻辑时，也能更方便地访问和修改这些配置项。
    retrieval_top_k: int
    retrieval_mode: str
    summary_strategy: str
    hybrid_candidate_multiplier: int
    hybrid_candidate_minimum: int
    embedding_provider: str
    embedding_model_name: str
    embedding_cache_dir: str | None
    reranker_provider: str
    reranker_model_name: str
    reranker_cache_dir: str | None
    

@dataclass
class Settings:
    llm_provider: str
    model_name: str
    api_key: str
    base_url: str | None
    course_source_root: str
    course_include_dirs: list[str] # 通过环境变量配置一些特定的子目录，来让检索更聚焦一些。比如只检索“课程内容”相关的文档，而不检索“项目实战”相关的文档。(扫描的文件目录白名单)
    vector_index_cache_path: str | None # 这个字段可以用来指定一个本地文件路径，用于缓存整个向量索引的数据结构。这样在后续的使用中，我们就可以直接从这个缓存文件中加载向量索引，而不需要重新计算所有 chunks 的向量表示，进一步提升启动速度。
    retrieval: RetrievalSettings
    retrieval_trace_path: str | None


def get_settings() -> Settings:
    llm_provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    model_name = os.getenv("MODEL_NAME", "gpt-4.1-mini").strip()
    retrieval_mode=os.getenv("RETRIEVAL_MODE", "chunk").strip().lower()
    summary_strategy = os.getenv("SUMMARY_STRATEGY", "same-source").strip().lower()
    course_source_root = os.getenv(
        "COURSE_SOURCE_ROOT",
        "/Users/a1-6/Desktop/AIAgent/code",
    ).strip()
    raw_include_dirs = os.getenv("COURSE_INCLUDE_DIRS", "1-3-SafeandReliableAIviaGuardrails,2-1-RetrievalAugmentedGeneration,2-2-BuildingAndEvaluatingAdvancedRAGApplications,2-3-ai-agents-for-beginners").strip()
    course_include_dirs = [item.strip() for item in raw_include_dirs.split(",") if item.strip()]
    retrieval_top_k = int(os.getenv("RETRIEVAL_TOP_K", "5"))
    hybrid_candidate_multiplier = int(os.getenv("HYBRID_CANDIDATE_MULTIPLIER", "3"))
    hybrid_candidate_minimum = int(os.getenv("HYBRID_CANDIDATE_MINIMUM", "10"))
    embedding_provider = os.getenv("EMBEDDING_PROVIDER", "hash").strip().lower()
    embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3").strip()
    embedding_cache_dir = os.getenv("EMBEDDING_CACHE_DIR", "").strip() or None
    vector_index_cache_path = os.getenv(
        "VECTOR_INDEX_CACHE_PATH",
        "/Users/a1-6/Desktop/AIAgent/05-project/course-tutor-agent/data/vector_index_cache.json",
    ).strip() or None
    reranker_provider = os.getenv("RERANKER_PROVIDER", "none").strip().lower()
    reranker_model_name = os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-base").strip()
    reranker_cache_dir = os.getenv("RERANKER_CACHE_DIR", "").strip() or embedding_cache_dir

    retrieval_trace_path = os.getenv(
        "RETRIEVAL_TRACE_PATH",
        "/Users/a1-6/Desktop/AIAgent/05-project/course-tutor-agent/data/retrieval_traces.jsonl",
    ).strip() or None

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
        course_include_dirs=course_include_dirs,
        vector_index_cache_path=vector_index_cache_path,
        retrieval=RetrievalSettings(
            retrieval_top_k=retrieval_top_k,
            retrieval_mode=retrieval_mode,
            summary_strategy=summary_strategy,
            hybrid_candidate_multiplier=hybrid_candidate_multiplier,
            hybrid_candidate_minimum=hybrid_candidate_minimum,
            embedding_provider=embedding_provider,
            embedding_model_name=embedding_model_name,
            embedding_cache_dir=embedding_cache_dir,
            reranker_provider=reranker_provider,
            reranker_model_name=reranker_model_name,
            reranker_cache_dir=reranker_cache_dir,
        ),
        retrieval_trace_path=retrieval_trace_path,
    )


def validate_settings(settings: Settings) -> None:
    if settings.llm_provider not in {"openai", "github"}:
        raise ValueError("LLM_PROVIDER must be either 'openai' or 'github'.")

    if not settings.api_key:
        if settings.llm_provider == "github":
            raise ValueError("GITHUB_TOKEN is required when LLM_PROVIDER=github.")
        raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai.")
    
    if settings.retrieval.retrieval_mode not in {"document", "chunk", "vector", "hybrid"}: # 目前支持三种检索模式：最原始的文档级检索（document），更细粒度的切块级检索（chunk），以及基于向量搜索的检索（vector）。用户可以根据自己的需求和数据规模来选择合适的检索模式。一般来说，chunk 模式在大多数情况下会有更好的性能和相关性，而 vector 模式则适用于需要处理非常大规模文本数据或者需要更复杂语义理解的场景。
        raise ValueError("RETRIEVAL_MODE must be 'document', 'chunk', 'vector', or 'hybrid'.")
    
    if settings.retrieval.embedding_provider not in {"hash", "sentence-transformers"}: # 目前先支持两种 embedding 提供商，后续如果需要支持更多，可以继续扩展这个条件判断。
        raise ValueError("EMBEDDING_PROVIDER must be 'hash' or 'sentence-transformers'.")
    
    if settings.retrieval.reranker_provider not in {"none", "sentence-transformers"}: # 目前先支持两种 reranker 提供商，后续如果需要支持更多，可以继续扩展这个条件判断。
        raise ValueError("RERANKER_PROVIDER must be 'none' or 'sentence-transformers'.")
    
    if settings.retrieval.hybrid_candidate_multiplier < 1: # 混合检索模式下，候选结果的数量应该至少是 top_k 的倍数，这样才能保证有足够的候选结果供 reranker 进行重新排序。如果这个值设置得太小，就可能导致 reranker 没有足够的候选结果来发挥作用，从而影响最终的检索效果。
        raise ValueError("HYBRID_CANDIDATE_MULTIPLIER must be >= 1.")
    
    if settings.retrieval.hybrid_candidate_minimum < 1: # 混合检索模式下，候选结果的数量应该有一个合理的下限，以确保在 top_k 较小的情况下，reranker 仍然有足够的候选结果来进行重新排序。如果这个值设置得太小，就可能导致在 top_k 较小的情况下，reranker 没有足够的候选结果来发挥作用，从而影响最终的检索效果。
        raise ValueError("HYBRID_CANDIDATE_MINIMUM must be >= 1.")
    
    if settings.retrieval.summary_strategy not in {"same-source"}: # 目前我们只实现了一种摘要策略 same-source，后续如果需要支持更多的摘要策略，可以继续扩展这个条件判断。
        raise ValueError("SUMMARY_STRATEGY must be 'same-source'.")

