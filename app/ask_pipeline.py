"""Shared grounded-ask pipeline used by /ask and OpenAI-compatible chat."""
from __future__ import annotations

from time import perf_counter
from typing import Any, Literal
from uuid import uuid4

from app.audit import write_ask_audit
from app.provenance import source_label
from app.rag import generate_answer, retrieve_evidence
from app.typed_atomic_sufficiency import assess_typed_atomic_sufficiency

INSUFFICIENT_EVIDENCE_MESSAGE = (
    "I don't have sufficient evidence in the "
    "retrieved sources to answer this question."
)


def to_evidence_source(source: dict) -> dict:
    return {
        "label": source_label(source),
        "filename": source["filename"],
        "chunk_index": source["chunk_index"],
        "page_number": source["page_number"],
        "content": source["content"],
        "excerpt": source["content"],
        "similarity": source["similarity"],
    }


def run_grounded_ask(
    question: str,
    *,
    model: str,
) -> dict[str, Any]:
    """
    Retrieve evidence, refuse when insufficient, otherwise generate.

    Returns a dict with request_id, status, question, answer,
    sources, sufficiency, and timings.
    """
    request_id = str(uuid4())
    total_started = perf_counter()

    retrieval_started = perf_counter()
    sources = retrieve_evidence(question)
    retrieval_ms = (perf_counter() - retrieval_started) * 1000

    sufficiency_started = perf_counter()
    sufficiency = assess_typed_atomic_sufficiency(question, sources)
    sufficiency_ms = (perf_counter() - sufficiency_started) * 1000

    response_sources = [to_evidence_source(source) for source in sources]
    source_labels = [source_label(source) for source in sources]

    status: Literal["answered", "insufficient_evidence"]
    answer: str
    generation_ms: float | None

    if not sufficiency.sufficient:
        status = "insufficient_evidence"
        answer = INSUFFICIENT_EVIDENCE_MESSAGE
        generation_ms = None
        total_ms = (perf_counter() - total_started) * 1000
    else:
        generation_started = perf_counter()
        answer = generate_answer(question, sources)
        generation_ms = (perf_counter() - generation_started) * 1000
        total_ms = (perf_counter() - total_started) * 1000
        status = "answered"

    write_ask_audit(
        request_id=request_id,
        question=question,
        status=status,
        answer=answer,
        sufficient=status == "answered",
        source_labels=source_labels,
        model=model,
        retrieval_ms=retrieval_ms,
        sufficiency_ms=sufficiency_ms,
        generation_ms=generation_ms,
        total_ms=total_ms,
    )

    return {
        "request_id": request_id,
        "status": status,
        "question": question,
        "answer": answer,
        "sources": response_sources,
        "sufficiency": {"sufficient": status == "answered"},
        "timings": {
            "retrieval_ms": retrieval_ms,
            "sufficiency_ms": sufficiency_ms,
            "generation_ms": generation_ms,
            "total_ms": total_ms,
        },
    }


def format_chat_answer(result: dict[str, Any]) -> str:
    """Answer text plus compact source footnotes for chat UIs."""
    answer = result["answer"]
    if result["status"] != "answered":
        return answer

    labels = [
        source["label"]
        for source in result["sources"]
        if source.get("label")
    ]
    if not labels:
        return answer

    unique_labels = list(dict.fromkeys(labels))
    footnotes = "\n".join(f"- {label}" for label in unique_labels)
    return f"{answer}\n\nSources:\n{footnotes}"
