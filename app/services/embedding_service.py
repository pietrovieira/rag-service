"""
EmbeddingService - abstração plugável.
Por padrão: Mock determinístico (hash) sem dependência externa.
Troque por SentenceTransformers/OpenAI implementando embed(texts: list[str]) -> list[list[float]]
"""
import hashlib
import math
from typing import Protocol


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...
    @property
    def dim(self) -> int: ...


class HashEmbedder:
    """Mock leve, determinístico e rápido. Ideal para demo/testes sem API key."""

    def __init__(self, dim: int = 384):
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def _text_to_vector(self, text: str) -> list[float]:
        # Gera vetor determinístico via hash
        vec = []
        for i in range(self.dim):
            h = hashlib.md5(f"{text}::{i}".encode()).digest()
            # 0..255 -> -1..1
            val = (int.from_bytes(h[:2], "big") / 65535.0) * 2 - 1
            vec.append(val)
        # normaliza L2
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._text_to_vector(t) for t in texts]


# Factory - futuro plugável
def get_embedder(dim: int = 384, kind: str = "mock") -> Embedder:
    if kind == "mock":
        return HashEmbedder(dim=dim)
    # Exemplo para plugar OpenAI:
    # if kind == "openai":
    #     return OpenAIEmbedder(dim=dim)
    # if kind == "sentence_transformers":
    #     return STEmbedder(...)
    return HashEmbedder(dim=dim)
