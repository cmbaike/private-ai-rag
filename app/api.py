"""Private AI RAG API — document ingest, search, and grounded ask."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import psycopg
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.ask_pipeline import (
    INSUFFICIENT_EVIDENCE_MESSAGE,
    run_grounded_ask,
)
from app.ingestion import ingest_document
from app.openai_compat import router as openai_compat_router
from app.rag import retrieve, VLLM_MODEL
from app.security.dependencies import require_api_key_principal

load_dotenv()

DATABASE_URL = os.environ["DATABASE_URL"]

app = FastAPI(
    title="Private AI RAG",
    description="Self-hosted evidence-grounded document Q&A",
    version="0.1.0",
)
app.include_router(openai_compat_router)


class DocumentSummary(BaseModel):
    id: int
    filename: str
    sha256: str | None
    content_type: str | None
    size_bytes: int | None
    status: Literal["ready"]
    chunks: int
    pages: int
    created_at: str


class DocumentDetail(DocumentSummary):
    first_page: int | None
    last_page: int | None


class DeleteDocumentResponse(BaseModel):
    status: Literal["deleted"]
    document_id: int
    filename: str


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    limit: int = Field(default=3, ge=1, le=10)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)


class AskTimings(BaseModel):
    retrieval_ms: float
    sufficiency_ms: float
    generation_ms: float | None
    total_ms: float


class Source(BaseModel):
    filename: str
    chunk_index: int
    page_number: int | None
    content: str
    similarity: float


class EvidenceSource(Source):
    label: str
    excerpt: str


class SufficiencySummary(BaseModel):
    sufficient: bool


class AskResponse(BaseModel):
    request_id: str
    status: Literal["answered", "insufficient_evidence"]
    question: str
    answer: str
    sources: list[EvidenceSource]
    sufficiency: SufficiencySummary
    timings: AskTimings


@app.get("/health")
def health():
    database = False
    try:
        with psycopg.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                database = cur.fetchone()[0] == 1
    except Exception:
        pass
    return {"status": "ok" if database else "degraded", "database": database}


@app.post("/search", response_model=list[Source], dependencies=[Depends(require_api_key_principal)])
def search(request: SearchRequest):
    try:
        return retrieve(request.query, request.limit)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from None


@app.post("/ask", response_model=AskResponse, dependencies=[Depends(require_api_key_principal)])
def ask(request: AskRequest):
    try:
        return run_grounded_ask(request.question, model=VLLM_MODEL)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from None


@app.post("/documents", dependencies=[Depends(require_api_key_principal)],)
async def upload_document(
    file: UploadFile = File(...),
):
    allowed_extensions = {
        ".txt",
        ".md",
        ".pdf",
    }

    extension = Path(file.filename or "").suffix.lower()

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported document type. "
                "Supported: TXT, MD, PDF."
            ),
        )

    try:
        data = await file.read()

        if not data:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty.",
            )

        result = ingest_document(
            filename=file.filename,
            content_type=(
                file.content_type
                or "application/octet-stream"
            ),
            data=data,
        )

        return result

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


@app.get(
    "/documents",
    response_model=list[DocumentSummary], dependencies=[Depends(require_api_key_principal)],
)
def list_documents():
    try:
        with psycopg.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        d.id,
                        d.filename,
                        d.sha256,
                        d.content_type,
                        d.size_bytes,
                        d.created_at,
                        COUNT(c.id) AS chunks,
                        COUNT(
                            DISTINCT c.page_number
                        ) AS pages
                    FROM documents d
                    LEFT JOIN chunks c
                        ON c.document_id = d.id
                    GROUP BY
                        d.id,
                        d.filename,
                        d.sha256,
                        d.content_type,
                        d.size_bytes,
                        d.created_at
                    ORDER BY d.created_at DESC
                    """
                )

                rows = cur.fetchall()

        return [
            {
                "id": row[0],
                "filename": row[1],
                "sha256": row[2],
                "content_type": row[3],
                "size_bytes": row[4],
                "created_at": row[5].isoformat(),
                "status": "ready",
                "chunks": row[6],
                "pages": row[7],
            }
            for row in rows
        ]

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


@app.get(
    "/documents/{document_id}",
    response_model=DocumentDetail, dependencies=[Depends(require_api_key_principal)],)
def get_document(document_id: int):
    try:
        with psycopg.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        d.id,
                        d.filename,
                        d.sha256,
                        d.content_type,
                        d.size_bytes,
                        d.created_at,
                        COUNT(c.id) AS chunks,
                        COUNT(
                            DISTINCT c.page_number
                        ) AS pages,
                        MIN(c.page_number) AS first_page,
                        MAX(c.page_number) AS last_page
                    FROM documents d
                    LEFT JOIN chunks c
                        ON c.document_id = d.id
                    WHERE d.id = %s
                    GROUP BY
                        d.id,
                        d.filename,
                        d.sha256,
                        d.content_type,
                        d.size_bytes,
                        d.created_at
                    """,
                    (document_id,),
                )

                row = cur.fetchone()

        if row is None:
            raise HTTPException(
                status_code=404,
                detail="Document not found.",
            )

        return {
            "id": row[0],
            "filename": row[1],
            "sha256": row[2],
            "content_type": row[3],
            "size_bytes": row[4],
            "created_at": row[5].isoformat(),
            "status": "ready",
            "chunks": row[6],
            "pages": row[7],
            "first_page": row[8],
            "last_page": row[9],
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


@app.delete("/documents/{document_id}", response_model=DeleteDocumentResponse,dependencies=[Depends(require_api_key_principal)],)
def delete_document(document_id: int):

    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:

            cur.execute(
                """
                DELETE FROM documents
                WHERE id = %s
                RETURNING filename
                """,
                (document_id,),
            )

            deleted = cur.fetchone()

        conn.commit()

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    return {
        "status": "deleted",
        "document_id": document_id,
        "filename": deleted[0],
    }
