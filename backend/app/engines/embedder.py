from __future__ import annotations

from functools import lru_cache
from hashlib import blake2b

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:
    SentenceTransformer = None  # type: ignore

from ..config import settings


class ChunkEmbedder:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.embedding_model
        self._model: SentenceTransformer | None = None

    def _load_model(self) -> SentenceTransformer:
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers is not installed")
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def dimension(self) -> int:
        try:
            return self._load_model().get_sentence_embedding_dimension()
        except Exception:
            return 8

    @staticmethod
    def _fallback_embedding(text: str, *, dimension: int = 8) -> list[float]:
        digest = blake2b(text.encode("utf-8"), digest_size=dimension).digest()
        return [byte / 255.0 for byte in digest]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            embeddings = self._load_model().encode(
                texts,
                batch_size=settings.embedding_batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except Exception:
            return [self._fallback_embedding(text) for text in texts]
        if hasattr(embeddings, "tolist"):
            return embeddings.tolist()
        return [list(vector) for vector in embeddings]


@lru_cache(maxsize=1)
def get_embedder() -> ChunkEmbedder:
    return ChunkEmbedder()

