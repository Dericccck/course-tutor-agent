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


def get_settings() -> Settings:
    llm_provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    model_name = os.getenv("MODEL_NAME", "gpt-4.1-mini").strip()
    course_source_root = os.getenv(
        "COURSE_SOURCE_ROOT",
        "/Users/a1-6/Desktop/AIAgent/code",
    ).strip()
    retrieval_top_k = int(os.getenv("RETRIEVAL_TOP_K", "5"))

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
    )


def validate_settings(settings: Settings) -> None:
    if settings.llm_provider not in {"openai", "github"}:
        raise ValueError("LLM_PROVIDER must be either 'openai' or 'github'.")

    if not settings.api_key:
        if settings.llm_provider == "github":
            raise ValueError("GITHUB_TOKEN is required when LLM_PROVIDER=github.")
        raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai.")

