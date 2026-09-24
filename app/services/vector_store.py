import json
import uuid
import math
from dataclasses import dataclass, field
from typing import Protocol

from app.config import get_settings


@dataclass
class Document:
    id: str
    text: str
    vector: list[float]
    metadata: dict = field(default_factory=dict)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1
    nb = math.sqrt(sum(y * y for y in b)) or 1
    return dot / (na * nb)


class VectorStore(Protocol):
    def add(self, texts: list[str], vectors: list[list[float]], metadatas: list[dict] | None = None) -> list[str]: ...
    def search(self, query_vector: list[float], top_k: int = 5) -> list[tuple[Document, float]]: ...
    def count(self) -> int: ...
    def clear(self) -> None: ...


class InMemoryVectorStore:
    """Vector store simples em memória com busca por cosine similarity."""

    def __init__(self):
        self._docs: dict[str, Document] = {}

    def add(self, texts: list[str], vectors: list[list[float]], metadatas: list[dict] | None = None) -> list[str]:
        ids = []
        for i, (text, vec) in enumerate(zip(texts, vectors)):
            doc_id = str(uuid.uuid4())
            meta = metadatas[i] if metadatas and i < len(metadatas) else {}
            self._docs[doc_id] = Document(id=doc_id, text=text, vector=vec, metadata=meta)
            ids.append(doc_id)
        return ids

    def search(self, query_vector: list[float], top_k: int = 5) -> list[tuple[Document, float]]:
        if not self._docs:
            return []
        scored = [(doc, cosine_similarity(query_vector, doc.vector)) for doc in self._docs.values()]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def count(self) -> int:
        return len(self._docs)

    def clear(self):
        self._docs.clear()

    def delete_all(self):
        self.clear()


class PgVectorStore:
    """Vector store persistente com Postgres + pgvector."""

    def __init__(self, database_url: str, embedding_dim: int = 384):
        self.database_url = database_url
        self.embedding_dim = embedding_dim
        self._ensure_schema()

    def _connect(self, register: bool = True):
        import psycopg

        conn = psycopg.connect(self.database_url)
        if register:
            from pgvector.psycopg import register_vector

            register_vector(conn)
        return conn

    def _ensure_schema(self):
        with self._connect(register=False) as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.commit()

        with self._connect(register=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS embeddings (
                        id UUID PRIMARY KEY,
                        text TEXT NOT NULL,
                        embedding vector({self.embedding_dim}) NOT NULL,
                        metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """
                )
            conn.commit()

        with self._connect(register=True) as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute(
                        """
                        CREATE INDEX IF NOT EXISTS embeddings_embedding_idx
                        ON embeddings USING hnsw (embedding vector_cosine_ops)
                        """
                    )
                    conn.commit()
                except Exception:
                    conn.rollback()

    def add(self, texts: list[str], vectors: list[list[float]], metadatas: list[dict] | None = None) -> list[str]:
        ids: list[str] = []
        with self._connect() as conn:
            with conn.cursor() as cur:
                for i, (text, vec) in enumerate(zip(texts, vectors)):
                    doc_id = str(uuid.uuid4())
                    meta = metadatas[i] if metadatas and i < len(metadatas) else {}
                    cur.execute(
                        """
                        INSERT INTO embeddings (id, text, embedding, metadata)
                        VALUES (%s::uuid, %s, %s, %s::jsonb)
                        """,
                        (doc_id, text, vec, json.dumps(meta)),
                    )
                    ids.append(doc_id)
            conn.commit()
        return ids

    def search(self, query_vector: list[float], top_k: int = 5) -> list[tuple[Document, float]]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id::text, text, embedding, metadata,
                           1 - (embedding <=> %s::vector) AS score
                    FROM embeddings
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (query_vector, query_vector, top_k),
                )
                rows = cur.fetchall()

        results: list[tuple[Document, float]] = []
        for doc_id, text, embedding, metadata, score in rows:
            meta = metadata if isinstance(metadata, dict) else json.loads(metadata or "{}")
            if embedding is None:
                vec: list[float] = []
            elif hasattr(embedding, "to_list"):
                vec = list(embedding.to_list())
            elif hasattr(embedding, "tolist"):
                vec = list(embedding.tolist())
            else:
                vec = [float(x) for x in embedding]
            results.append(
                (
                    Document(id=doc_id, text=text, vector=vec, metadata=meta),
                    float(score),
                )
            )
        return results

    def count(self) -> int:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM embeddings")
                return int(cur.fetchone()[0])

    def clear(self):
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("TRUNCATE TABLE embeddings")
            conn.commit()

    def delete_all(self):
        self.clear()


_vector_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        settings = get_settings()
        if settings.vector_db_type == "pgvector":
            _vector_store = PgVectorStore(
                database_url=settings.database_url,
                embedding_dim=settings.embedding_dim,
            )
        else:
            _vector_store = InMemoryVectorStore()
    return _vector_store


def reset_vector_store() -> None:
    """Útil em testes para recriar o singleton."""
    global _vector_store
    _vector_store = None
