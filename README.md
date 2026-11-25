# Private AI RAG

Self-hosted, evidence-grounded document Q&A for operations teams.

Retrieve approved docs, refuse when evidence is insufficient, and return
cited answers. The HTTP API is the product; Open WebUI is an optional
chat front end.

## Capabilities

| Area | What you get |
|------|----------------|
| Ask | `/ask` with typed sufficiency; refuse without enough evidence |
| Chat UI | Open WebUI (ChatGPT-style) via OpenAI-compatible `/v1` |
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

### Open WebUI (ChatGPT-style chat)

The API exposes OpenAI-compatible `/v1/models` and `/v1/chat/completions`
that run the same grounded ask pipeline (retrieve → sufficiency → answer
or refuse). Open WebUI connects to those endpoints.

```bash
# Terminal 1 — RAG API (as above)
uv run uvicorn app.api:app --host 127.0.0.1 --port 8081

# Terminal 2 — Open WebUI
export OPEN_WEBUI_OPENAI_API_KEY='your-raw-api-key'  # must match a digest in PRIVATE_AI_API_KEYS_JSON
docker compose -f docker-compose.openwebui.yml up -d
```

Open `http://127.0.0.1:3000`, create the admin account if prompted, then
chat using the configured model. Auth accepts `Authorization: Bearer`
(Open WebUI) or `X-API-Key`.

Copy `.env.example` to `.env` for local configuration. Do not commit `.env`
or raw API keys.

## Development

```bash
uv sync --group dev
uv run pytest
```

## License

[MIT](LICENSE) © 2026 Christopher Mbaike
