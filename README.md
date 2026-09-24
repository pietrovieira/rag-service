# RAG Service — FastAPI + pgvector

Microserviço de **RAG** (Retrieval-Augmented Generation) com FastAPI, embeddings e armazenamento vetorial em **Postgres + pgvector**.

Clientes (Rails, React/Next, mobile) enviam um prompt para `POST /query`. O serviço vetoriza a pergunta, busca os top-k chunks no pgvector, monta o contexto e chama o LLM.

---

## Arquitetura

![Arquitetura RAG — FastAPI • Fluxo de Consulta e Ingestão](docs/rag-fastapi-fluxo.png)

### Fluxo de consulta (online)

1. **Apps** enviam um `prompt` para o FastAPI (`POST /query` + orquestração)
2. **Modelo de embedding** vetoriza a query (`text-embedding-3` / E5 / mock local)
3. **Vector DB (pgvector)** faz busca top-k por similaridade
4. **LLM** (GPT-4o / Claude / Llama / mock) recebe prompt + contexto
5. Retorna **Response** (JSON / Markdown) para o cliente

### Fluxo de ingestão (offline)

Executado antes das consultas (e de forma incremental quando há novos documentos):

```
Documentos (PDF / sites / docs / textarea)
    → Chunking (500–1000 tokens, com overlap)
    → Embedding (vetoriza cada chunk)
    → Vector DB — pgvector (free)
```

> Após a busca, a LLM analisa primeiro os documentos recuperados; se não encontrar informação suficiente, o stub atual indica isso na resposta (fontes externas ficam como extensão futura).

---

## Stack

| Camada | Tecnologia |
|--------|------------|
| API | FastAPI + Uvicorn |
| Config | pydantic-settings (`.env` / `.env.local`) |
| Embedding | `HashEmbedder` mock (plugável: OpenAI / E5) |
| Vector DB | **pgvector** (Postgres 16) — fallback `in_memory` |
| LLM | `MockLLM` (plugável: OpenAI) |
| UI ingestão | `GET /documents` — textarea (1 doc por linha) |
| Container | Docker Compose (`rag-api` + `db`) |

---

## Estrutura do projeto

```
rag-service/
├── app/
│   ├── main.py                 # FastAPI app, CORS, routers
│   ├── config.py               # Settings (.env / .env.local)
│   ├── models.py               # Schemas Pydantic
│   ├── core/exceptions.py
│   ├── routers/
│   │   ├── health.py           # GET /, GET /health
│   │   ├── documents.py        # GET/POST /documents (+ API JSON)
│   │   ├── ingest.py           # POST/DELETE /ingest
│   │   └── query.py            # POST /query
│   └── services/
│       ├── embedding_service.py
│       ├── vector_store.py     # InMemory + PgVectorStore
│       ├── llm_service.py
│       └── rag_service.py      # chunking + orquestração
├── docs/
│   └── rag-fastapi-fluxo.png   # Diagrama de arquitetura
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .env.example
└── README.md
```

---

## Pré-requisitos

- Python 3.11+
- Docker / Docker Compose (para Postgres + pgvector)
- (Opcional) chave OpenAI se `LLM_TYPE=openai`

---

## Setup rápido

### 1. Clone

```bash
gh repo clone pietrovieira/rag-service
cd rag-service
```

### 2. Ambiente

```bash
cp .env.example .env.local
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Variáveis principais em `.env.local`:

| Variável | Default | Descrição |
|----------|---------|-----------|
| `PORT` | `9876` | Porta HTTP da API |
| `VECTOR_DB_TYPE` | `pgvector` | `pgvector` ou `in_memory` |
| `DATABASE_URL` | `postgresql://rag:rag@localhost:5432/rag` | Conexão Postgres |
| `EMBEDDING_DIM` | `384` | Dimensão dos vetores |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `500` / `50` | Chunking |
| `TOP_K` | `5` | Chunks recuperados na query |
| `LLM_TYPE` | `mock` | `mock` ou `openai` |

### 3. Banco (pgvector)

```bash
docker compose up -d db
```

Sobe `pgvector/pgvector:pg16` na porta `5432`. Na primeira conexão a API cria a extensão `vector` e a tabela `embeddings`.

### 4. API

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 9876
```

- UI de documentos: http://localhost:9876/documents  
- Swagger: http://localhost:9876/docs  
- Health: http://localhost:9876/health  

### 5. Stack completa (API + DB)

```bash
docker compose up --build
```

A API fica em **http://localhost:9876**.

---

## Endpoints

| Método | Rota | Descrição |
|--------|------|-----------|
| `GET` | `/` | Info do serviço |
| `GET` | `/health` | Status + contagem de embeddings |
| `GET` | `/documents` | UI com textarea (1 documento por linha) |
| `POST` | `/documents` | Form: indexa linhas → chunks → embeddings → pgvector |
| `POST` | `/documents/ingest` | Mesmo fluxo via JSON (`IngestRequest`) |
| `POST` | `/ingest` | API de ingestão JSON |
| `DELETE` | `/ingest` | Limpa o vector store |
| `POST` | `/query` | RAG: pergunta → top-k → LLM |
| `GET` | `/docs` | OpenAPI / Swagger |

---

## Exemplos

### Ingestão pela UI

1. Abra http://localhost:9876/documents  
2. Cole um documento por linha no textarea  
3. Clique em **Indexar no pgvector**

### Ingestão via API

```bash
curl -X POST http://localhost:9876/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      "RAG combina recuperação vetorial com geração. O FastAPI orquestra a busca top-k no vector DB e envia contexto ao LLM.",
      "pgvector armazena embeddings no Postgres para busca por similaridade (cosine)."
    ]
  }'
```

### Query RAG

```bash
curl -X POST http://localhost:9876/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "O que é RAG?",
    "top_k": 3,
    "include_context": true
  }'
```

Resposta típica:

```json
{
  "answer": "...",
  "question": "O que é RAG?",
  "retrieved_chunks": [
    { "id": "...", "text": "...", "score": 0.91, "metadata": { "doc_index": 0, "chunk_index": 0 } }
  ],
  "latency_ms": 12.4
}
```

---

## Como o pgvector é usado

Tabela criada automaticamente:

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE embeddings (
  id UUID PRIMARY KEY,
  text TEXT NOT NULL,
  embedding vector(384) NOT NULL,
  metadata JSONB NOT NULL DEFAULT '{}',
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- índice HNSW (cosine)
CREATE INDEX embeddings_embedding_idx
  ON embeddings USING hnsw (embedding vector_cosine_ops);
```

Busca (similaridade de cosseno via distância `<=>`):

```sql
SELECT id, text, metadata,
       1 - (embedding <=> $1::vector) AS score
FROM embeddings
ORDER BY embedding <=> $1::vector
LIMIT $2;
```

Implementação: `app/services/vector_store.py` → `PgVectorStore`.

---

## Plugando serviços reais

- **Embedding OpenAI / E5**: implemente em `embedding_service.py` e ajuste `get_embedder()`
- **LLM OpenAI**: `LLM_TYPE=openai` + `OPENAI_API_KEY` em `.env.local` (e `OpenAILLM` em `llm_service.py`)
- **Só memória (sem Postgres)**: `VECTOR_DB_TYPE=in_memory`

---

## Desenvolvimento

```bash
# testes
pytest -q

# health
curl http://localhost:9876/health
```

---

## Licença

Uso interno / demonstração do fluxo RAG com FastAPI e pgvector.
