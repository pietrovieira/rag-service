from fastapi import APIRouter
from app.models import IngestRequest, IngestResponse
from app.services.rag_service import get_rag_service

router = APIRouter(prefix="/ingest", tags=["ingest"])


@router.post("", response_model=IngestResponse, summary="Indexa documentos (chunk + embedding + vector DB)")
def ingest(req: IngestRequest):
    rag = get_rag_service()
    inserted = rag.ingest(
        documents=req.documents,
        metadatas=req.metadatas,
        chunk_size=req.chunk_size,
        chunk_overlap=req.chunk_overlap,
    )
    return IngestResponse(inserted_chunks=inserted, message=f"{inserted} chunks indexados com sucesso")


@router.delete("", summary="Limpa vector store (útil para testes)")
def clear():
    from app.services.vector_store import get_vector_store

    get_vector_store().clear()
    return {"message": "vector store limpo"}
