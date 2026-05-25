from pydantic_settings import BaseSettings, SettingsConfigDict


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

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
