from urllib.parse import quote

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from app.models import IngestRequest, IngestResponse
from app.services.rag_service import get_rag_service
from app.services.vector_store import get_vector_store

router = APIRouter(tags=["documents"])


def _documents_page(message: str | None = None, error: str | None = None) -> str:
    banner = ""
    if message:
        banner = f'<p class="ok">{message}</p>'
    elif error:
        banner = f'<p class="err">{error}</p>'

    try:
        count = get_vector_store().count()
    except Exception:
        count = "?"

    return f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Documentos — RAG</title>
  <style>
    :root {{
      --bg: #0f1419;
      --panel: #1a2332;
      --text: #e7ecf3;
      --muted: #8b9bb4;
      --accent: #3d8bfd;
      --ok: #3dd68c;
      --err: #f07178;
      --border: #2a3548;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      background:
        radial-gradient(ellipse at top left, #1a2a44 0%, transparent 50%),
        radial-gradient(ellipse at bottom right, #1a2838 0%, transparent 45%),
        var(--bg);
      color: var(--text);
    }}
    main {{
      max-width: 720px;
      margin: 0 auto;
      padding: 2.5rem 1.25rem 4rem;
    }}
    h1 {{
      font-size: 1.75rem;
      font-weight: 600;
      margin: 0 0 0.35rem;
      letter-spacing: -0.02em;
    }}
    .sub {{
      color: var(--muted);
      margin: 0 0 1.75rem;
      line-height: 1.5;
      font-size: 0.95rem;
    }}
    form {{
      display: flex;
      flex-direction: column;
      gap: 1rem;
    }}
    label {{
      font-size: 0.85rem;
      color: var(--muted);
      margin-bottom: -0.5rem;
    }}
    textarea {{
      width: 100%;
      min-height: 280px;
      padding: 1rem 1.1rem;
      border: 1px solid var(--border);
      border-radius: 10px;
      background: var(--panel);
      color: var(--text);
      font-family: "IBM Plex Mono", ui-monospace, monospace;
      font-size: 0.9rem;
      line-height: 1.55;
      resize: vertical;
    }}
    textarea:focus {{
      outline: 2px solid var(--accent);
      outline-offset: 1px;
    }}
    .row {{
      display: flex;
      align-items: center;
      gap: 1rem;
      flex-wrap: wrap;
    }}
    button {{
      appearance: none;
      border: none;
      background: var(--accent);
      color: #fff;
      font-weight: 600;
      font-size: 0.95rem;
      padding: 0.7rem 1.25rem;
      border-radius: 8px;
      cursor: pointer;
    }}
    button:hover {{ filter: brightness(1.08); }}
    .meta {{
      color: var(--muted);
      font-size: 0.85rem;
    }}
    .ok {{ color: var(--ok); margin: 0 0 1rem; }}
    .err {{ color: var(--err); margin: 0 0 1rem; }}
    a {{ color: var(--accent); }}
  </style>
</head>
<body>
  <main>
    <h1>Documentos</h1>
    <p class="sub">
      Digite um documento por linha. Ao enviar, cada linha vira chunks + embeddings
      e é salva no Postgres com <strong>pgvector</strong>.
    </p>
    {banner}
    <form method="post" action="/documents">
      <label for="documents">Documentos (1 por linha)</label>
      <textarea
        id="documents"
        name="documents"
        autofocus
        placeholder="RAG combina recuperação vetorial com geração de texto.&#10;pgvector armazena embeddings no Postgres para busca por similaridade.&#10;FastAPI orquestra ingestão, busca top-k e resposta do LLM."
      ></textarea>
      <div class="row">
        <button type="submit">Indexar no pgvector</button>
        <span class="meta">Embeddings no banco: <strong>{count}</strong> · <a href="/docs">API</a> · <a href="/health">health</a></span>
      </div>
    </form>
  </main>
</body>
</html>
"""


@router.get("/documents", response_class=HTMLResponse, summary="UI para colar documentos (1 por linha)")
def documents_form(request: Request):
    message = request.query_params.get("msg")
    error = request.query_params.get("err")
    return HTMLResponse(_documents_page(message=message, error=error))


@router.post("/documents", response_class=HTMLResponse, summary="Indexa linhas do textarea no pgvector")
def documents_submit(documents: str = Form("")):
    lines = [line.strip() for line in documents.splitlines() if line.strip()]
    if not lines:
        return RedirectResponse(url="/documents?err=Nenhuma+linha+preenchida", status_code=303)

    try:
        rag = get_rag_service()
        inserted = rag.ingest(documents=lines)
        return RedirectResponse(
            url=f"/documents?msg={inserted}+chunks+indexados+no+pgvector",
            status_code=303,
        )
    except Exception as exc:
        return RedirectResponse(url=f"/documents?err={quote(str(exc))}", status_code=303)


@router.post(
    "/documents/ingest",
    response_model=IngestResponse,
    summary="API JSON: lista de textos → chunks + embeddings no pgvector",
)
def documents_ingest_api(req: IngestRequest):
    rag = get_rag_service()
    inserted = rag.ingest(
        documents=req.documents,
        metadatas=req.metadatas,
        chunk_size=req.chunk_size,
        chunk_overlap=req.chunk_overlap,
    )
    return IngestResponse(inserted_chunks=inserted, message=f"{inserted} chunks indexados no pgvector")
