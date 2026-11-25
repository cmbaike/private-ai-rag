# Private AI RAG

Self-hosted, evidence-grounded document Q&A for operations teams.

Retrieve approved docs, refuse when evidence is insufficient, and return
cited answers. This repository is the **RAG** half of the former
`private-ai-platform` monorepo. Kubernetes investigations live in
[`private-ai-k8s-investigate`](../private-ai-k8s-investigate).

## Capabilities

| Area | What you get |
|------|----------------|
| Ask | `/ask` with typed sufficiency; refuse without enough evidence |
| Search | `/search` over ingested chunks (pgvector) |
| Documents | Upload / list / get / delete |
| Audit | Ask latency and source labels persisted |

## Quick start

```bash
cp .env.example .env   # set DATABASE_URL, PRIVATE_AI_API_KEYS_JSON, vLLM URL
uv sync
uv run python scripts/migrate.py
uv run uvicorn app.api:app --host 127.0.0.1 --port 8081
```

Interactive docs: `http://127.0.0.1:8081/docs`

## Related

- Kubernetes investigate core: `../private-ai-k8s-investigate`
- Split notes: see `SPLIT.md` in the original monorepo
