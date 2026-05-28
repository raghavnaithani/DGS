from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


class JobBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class IngestionRequest(JobBaseModel):
    query: str | None = Field(default=None, max_length=500)
    urls: list[HttpUrl] = Field(default_factory=list)

    @model_validator(mode="after")
    def _require_exactly_one_source(self) -> IngestionRequest:
        has_query = bool(self.query and self.query.strip())
        has_urls = bool(self.urls)
        if has_query == has_urls:
            raise ValueError("Provide exactly one of query or urls")
        return self


class JobSubmission(JobBaseModel):
    job_id: str
    status: Literal["queued"]


class JobRecord(JobBaseModel):
    id: str
    job_type: Literal["ingestion", "simulation"]
    request: dict[str, object]
    status: Literal["queued", "running", "completed", "failed"]
    progress: int = Field(ge=0, le=100)
    current_step: str
    total_sources: int = Field(ge=0)
    scraped_sources: int = Field(ge=0)
    stored_chunks: int = Field(ge=0)
    result: dict[str, object] | None = None
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime
