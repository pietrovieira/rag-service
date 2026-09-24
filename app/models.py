from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


# --- Ingest ---
class IngestRequest(BaseModel):
    documents: list[str] = Field(..., description="Lista de textos/documentos para indexar")
    metadatas: Optional[list[dict]] = Field(None, description="Metadados opcionais por documento")
    chunk_size: Optional[int] = Field(None, description="Override chunk_size")
    chunk_overlap: Optional[int] = Field(None, description="Override chunk_overlap")


class IngestResponse(BaseModel):
    inserted_chunks: int
    message: str


# --- Query (RAG) ---
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, examples=["O que é RAG?"])
    top_k: Optional[int] = Field(None, ge=1, le=20, description="Quantos chunks recuperar")
    include_context: bool = Field(False, description="Retornar chunks recuperados")


class RetrievedChunk(BaseModel):
    id: str
    text: str
    score: float
    metadata: dict


class QueryResponse(BaseModel):
    answer: str
    question: str
    retrieved_chunks: Optional[list[RetrievedChunk]] = None
    latency_ms: float


# --- Health ---
class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: datetime
    vector_store_docs: int
