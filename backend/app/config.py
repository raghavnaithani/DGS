try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except Exception:
    BaseSettings = object
    SettingsConfigDict = dict


class Settings(BaseSettings):
    groq_api_key: str = ""
    groq_model: str = "llama-3.1-8b-instant"
    disable_live_scraping: bool = False
    lancedb_path: str = ".dgs_lancedb"
    embedding_model: str = "ibm-granite/granite-embedding-30m-english"
    embedding_batch_size: int = 16
    ingestion_max_workers: int = 6
    scrape_max_workers: int = 6
    embedding_max_workers: int = 2
    search_max_results: int = 30
    retrieval_dense_limit: int = 20
    retrieval_bm25_limit: int = 20
    retrieval_similarity_threshold: float = 0.7
    retrieval_rrf_k: int = 60
    retrieval_enable_reranking: bool = True
    retrieval_rrf_dense_weight: float = 1.0
    retrieval_rrf_bm25_weight: float = 1.0
    retrieval_rrf_top_rank_bonus: float = 0.05
    retrieval_bm25_skip_threshold: float = 0.9
    retrieval_bm25_skip_gap: float = 0.2
    retrieval_expand_parents: bool = True
    search_page_delay_seconds: float = 0.25
    # Scraping safety knobs
    scrape_timeout_seconds: float = 30.0
    # Fraction of sources that may fail before the job aborts early (0.0-1.0)
    scrape_fail_fast_ratio: float = 0.5
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    sentry_dsn: str = ""
    supabase_url: str = ""
    supabase_key: str = ""
    debug: bool = False
    # Simulation / reasoning settings
    simulation_max_retries: int = 3
    simulation_temperature: float = 0.7
    simulation_max_tokens: int = 1024
    simulation_retry_penalty: float = 0.1
    simulation_worker_poll_interval_seconds: float = 0.25
    simulation_use_nli_grounding: bool = False

    try:
        model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")
    except Exception:
        model_config = {}


settings = Settings()
