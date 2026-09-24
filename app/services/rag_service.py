import time
import re
from .embedding_service import get_embedder
from .vector_store import get_vector_store
from .llm_service import get_llm
from app.config import get_settings
from app.models import RetrievedChunk


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Chunking simples por caracteres com overlap. Troque por tiktoken/langchain se precisar."""
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        # tenta quebrar em frase/palavra
        if end < len(text) and not chunk.endswith((" ", "\n", ".", "!", "?")):
            # procura último espaço
            last_space = chunk.rfind(" ")
            if last_space > chunk_size * 0.7:
                chunk = chunk[:last_space]
                end = start + last_space
        chunks.append(chunk.strip())
        if end >= len(text):
            break
        start = end - overlap
    return [c for c in chunks if c.strip()]


class RAGService:
    def __init__(self):
        self.settings = get_settings()
        self.embedder = get_embedder(dim=self.settings.embedding_dim, kind="mock")
        self.vdb = get_vector_store()
        self.llm = get_llm(kind=self.settings.llm_type, api_key=self.settings.openai_api_key, model=self.settings.openai_model)

    def ingest(self, documents: list[str], metadatas: list[dict] | None = None, chunk_size: int | None = None, chunk_overlap: int | None = None) -> int:
        cs = chunk_size or self.settings.chunk_size
        co = chunk_overlap or self.settings.chunk_overlap
        all_chunks: list[str] = []
        all_metas: list[dict] = []
        for idx, doc in enumerate(documents):
            chunks = chunk_text(doc, cs, co)
            for c_idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                base_meta = (metadatas[idx] if metadatas and idx < len(metadatas) else {})
                all_metas.append({**base_meta, "doc_index": idx, "chunk_index": c_idx})
        if not all_chunks:
            return 0
        vectors = self.embedder.embed(all_chunks)
        self.vdb.add(all_chunks, vectors, all_metas)
        return len(all_chunks)

    def query(self, question: str, top_k: int | None = None, include_context: bool = False) -> tuple[str, list[RetrievedChunk] | None, float]:
        start = time.time()
        k = top_k or self.settings.top_k
        q_vec = self.embedder.embed([question])[0]
        results = self.vdb.search(q_vec, top_k=k)
        if not results:
            answer = self.llm.generate(f"Contexto: \nPergunta: {question}")
            latency = (time.time() - start) * 1000
            return answer, [], latency

        # monta contexto
        context_blocks = []
        retrieved = []
        for doc, score in results:
            context_blocks.append(f"[{doc.id[:8]} | score={score:.3f}] {doc.text}")
            retrieved.append(RetrievedChunk(id=doc.id, text=doc.text, score=score, metadata=doc.metadata))

        context = "\n\n".join(context_blocks)
        prompt = f"Contexto:\n{context}\n\nPergunta: {question}\n\nResponda usando APENAS o contexto acima. Se não houver informação, diga que não encontrou."
        answer = self.llm.generate(prompt)
        latency = (time.time() - start) * 1000
        return answer, (retrieved if include_context else None), latency


# Singleton
_rag: RAGService | None = None

def get_rag_service() -> RAGService:
    global _rag
    if _rag is None:
        _rag = RAGService()
    return _rag
