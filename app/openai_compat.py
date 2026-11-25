"""OpenAI-compatible /v1 routes for Open WebUI and similar clients."""
from __future__ import annotations

import json
import time
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.ask_pipeline import format_chat_answer, run_grounded_ask
from app.rag import VLLM_MODEL
from app.security.dependencies import require_api_key_principal

router = APIRouter(prefix="/v1", tags=["openai-compatible"])


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None


def _last_user_question(messages: list[ChatMessage]) -> str:
    for message in reversed(messages):
        if message.role == "user" and isinstance(message.content, str):
            question = message.content.strip()
            if question:
                return question
    raise HTTPException(
        status_code=400,
        detail="At least one non-empty user message is required.",
    )


def _completion_payload(
    *,
    completion_id: str,
    created: int,
    model: str,
    content: str,
) -> dict[str, Any]:
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


def _stream_chunks(
    *,
    completion_id: str,
    created: int,
    model: str,
    content: str,
):
    first = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": content},
                "finish_reason": None,
            }
        ],
    }
    yield f"data: {json.dumps(first)}\n\n"

    done = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }
        ],
    }
    yield f"data: {json.dumps(done)}\n\n"
    yield "data: [DONE]\n\n"


@router.get("/models", dependencies=[Depends(require_api_key_principal)])
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": VLLM_MODEL,
                "object": "model",
                "created": 0,
                "owned_by": "private-ai-rag",
            }
        ],
    }


@router.post(
    "/chat/completions",
    dependencies=[Depends(require_api_key_principal)],
)
def chat_completions(request: ChatCompletionRequest):
    question = _last_user_question(request.messages)
    model = request.model or VLLM_MODEL

    try:
        result = run_grounded_ask(question, model=model)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from None

    content = format_chat_answer(result)
    completion_id = f"chatcmpl-{result['request_id']}"
    created = int(time.time())

    if request.stream:
        return StreamingResponse(
            _stream_chunks(
                completion_id=completion_id,
                created=created,
                model=model,
                content=content,
            ),
            media_type="text/event-stream",
        )

    return _completion_payload(
        completion_id=completion_id,
        created=created,
        model=model,
        content=content,
    )
