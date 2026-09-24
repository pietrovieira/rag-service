from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import get_settings
from app.routers import documents, health, ingest, query

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="""
Microserviço RAG com FastAPI

**Fluxo:**
1. `GET /documents` → textarea (1 doc por linha) → chunks + embeddings → **pgvector**
2. `POST /ingest` → chunking → embedding → Vector DB
3. `POST /query` → vetoriza pergunta → busca top-k → monta prompt → LLM → resposta

Docs: `/docs` (Swagger) e `/redoc`
    """,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# CORS liberado para demo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(documents.router)
app.include_router(ingest.router)
app.include_router(query.router)


# Para rodar: uvicorn app.main:app --reload
