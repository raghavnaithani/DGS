from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


class KnowledgeBaseModel(BaseModel):
	model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class SearchCandidate(KnowledgeBaseModel):
	title: str = Field(min_length=1)
	url: HttpUrl
	snippet: str = Field(default="")
	domain: str = Field(min_length=1)
	published_at: str | None = None
	score: float = 0.0


class ScrapedPage(KnowledgeBaseModel):
	title: str = Field(min_length=1)
	url: HttpUrl
	domain: str = Field(min_length=1)
	markdown: str = Field(default="")
	status: Literal["success", "failed", "skipped"] = "success"
	error_message: str | None = None


class ChunkDocument(KnowledgeBaseModel):
	id: str = Field(min_length=1)
	content: str = Field(min_length=1)
	source_url: str = Field(min_length=1)
	source_title: str | None = None
	chunk_index: int = Field(ge=0)
	parent_id: str | None = None
	parent_content: str | None = None
	section_title: str | None = None
	embedding: list[float] = Field(default_factory=list)
	created_at: datetime
	ttl_days: int = Field(default=30, ge=1)
	verification_status: Literal["verified", "unverified", "failed"] = "unverified"
	similarity_score: float | None = Field(default=None, ge=0.0, le=1.0)

	@model_validator(mode="after")
	def _validate_parent_content(self) -> ChunkDocument:
		if self.parent_id and not self.parent_content:
			raise ValueError("parent_content is required when parent_id is set")
		return self


class RetrievalRequest(KnowledgeBaseModel):
	query: str = Field(min_length=1, max_length=500)
	top_k: int = Field(default=10, ge=1, le=50)


class RetrievedEvidenceChunk(KnowledgeBaseModel):
	id: str = Field(min_length=1)
	content: str = Field(min_length=1)
	source_url: str = Field(min_length=1)
	source_title: str | None = None
	chunk_index: int = Field(ge=0)
	parent_id: str | None = None
	parent_content: str | None = None
	section_title: str | None = None
	citation: str = Field(min_length=1)
	rrf_score: float = Field(ge=0.0)
	dense_similarity: float | None = Field(default=None, ge=0.0, le=1.0)
	bm25_score: float | None = None
	context_type: str = "evidence"


class RetrievalResponse(KnowledgeBaseModel):
	query: str
	top_k: int = Field(ge=1, le=50)
	evidence: list[RetrievedEvidenceChunk] = Field(default_factory=list)
	expanded_context: list[RetrievedEvidenceChunk] = Field(default_factory=list)


from .schemas import KnowledgeChunk

