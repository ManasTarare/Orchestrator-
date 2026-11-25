"""Embedding interfaces and an optional Sentence Transformers adapter."""

from __future__ import annotations

from typing import Protocol, Sequence


class TextEmbedder(Protocol):
    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """Return one numeric vector for each input text."""
        ...


class SentenceTransformerEmbedder:
    """Load a caller-selected sentence embedding model.

    No model is selected by default. Record the selected model and its license
    before using this adapter outside local development.
    """

    def __init__(self, model_id: str, *, device: str | None = None) -> None:
        if not model_id.strip():
            raise ValueError("model_id is required")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError(
                "Install the routing extra with: pip install -e '.[routing]'"
            ) from error
        self.model_id = model_id
        self.model = SentenceTransformer(model_id, device=device)

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        vectors = self.model.encode(
            list(texts),
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vectors.tolist()
