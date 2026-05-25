from __future__ import annotations

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from ..config import settings


class ChunkEmbedder:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.embedding_model
        self._model: SentenceTransformer | None = None

    def _load_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def dimension(self) -> int:
        return self._load_model().get_sentence_embedding_dimension()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings = self._load_model().encode(
            texts,
            batch_size=settings.embedding_batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()


@lru_cache(maxsize=1)
def get_embedder() -> ChunkEmbedder:
    return ChunkEmbedder()

