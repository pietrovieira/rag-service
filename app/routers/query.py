from fastapi import APIRouter
from app.models import QueryRequest, QueryResponse
from app.services.rag_service import get_rag_service

router = APIRouter(prefix="/query", tags=["query"])


@router.post("", response_model=QueryResponse, summary="RAG: recupera contexto + gera resposta")
def query(req: QueryRequest):
    rag = get_rag_service()
    answer, retrieved, latency = rag.query(
        question=req.question,
        top_k=req.top_k,
        include_context=req.include_context,
    )
    return QueryResponse(
        answer=answer,
        question=req.question,
        retrieved_chunks=retrieved,
        latency_ms=round(latency, 2),
    )
