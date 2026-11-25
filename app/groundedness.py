import json
import os
import re

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from sympy import content


load_dotenv()

client = OpenAI(
    base_url=os.getenv(
        "GROUNDEDNESS_JUDGE_BASE_URL",
        "http://localhost:11434/v1",
    ),
    api_key=os.getenv(
        "GROUNDEDNESS_JUDGE_API_KEY",
        "ollama",
    ),
)

MODEL = os.getenv(
    "GROUNDEDNESS_JUDGE_MODEL",
    "gpt-oss:20b",
)


class GroundednessResult(BaseModel):
    grounded: bool
    reason: str
    judge_error: bool = False
    finish_reason: str | None = None
    completion_tokens: int | None = None


SYSTEM_PROMPT = """
You are a strict evidence-grounding judge.

Determine whether every material factual assertion in the CLAIM is
supported by the supplied EVIDENCE.

You are judging textual evidence support, not whether the claim is
technically true in the real world.

Rules:

1. Use only information explicitly contained in the supplied evidence.

2. Do not use outside knowledge, domain knowledge, common knowledge,
   background knowledge, or facts learned during training.

3. A claim is NOT grounded merely because it can be inferred using
   outside knowledge.

4. A paraphrase is grounded only when it preserves information already
   expressed by the evidence without introducing a new factual
   proposition.

5. Do not infer operational consequences from technical state.
   For example:
   - "all primary shards are assigned" does NOT by itself support
     "read operations remain functional"
   - "a container restarted" does NOT by itself support a particular
     restart cause
   - "a replica exists" does NOT by itself support availability,
     durability, or failover behavior

6. Do not infer causes, consequences, guarantees, exceptions, timings,
   recommendations, permissions, availability, safety properties, or
   operational behavior unless the evidence explicitly states them.

7. If the claim introduces a factual assertion that is absent from the
   evidence, return grounded=false even if that assertion is likely,
   reasonable, technically correct, or normally true.

8. Examine every material factual proposition in the claim separately.
   For each proposition, identify what text in the evidence supports it.

9. If any material proposition has no support in the evidence, return
   grounded=false. Do not ignore an unsupported proposition merely
   because another part of the claim is supported.

10. Subjects, actors, causes, sources, and scope must be supported.
    Evidence that establishes an event does not establish who or what
    caused that event. Evidence that establishes a fact does not
    establish that the fact applies to an additional product,
    environment, deployment, incident, system, condition, or time
    period introduced by the claim.
    
11. Before returning grounded=true, check whether removing any phrase
    from the claim would remove information that is absent from the
    evidence. If so, the complete claim is not grounded.
    
12. Return grounded=true only when every material factual proposition
    in the complete claim is supported by the supplied evidence.     

Return JSON only:

{"grounded": true, "reason": "brief explanation"}

or

{"grounded": false, "reason": "brief explanation"}
""".strip()


def parse_groundedness_response(
    content: str,
    finish_reason: str | None,
    completion_tokens: int | None = None,
) -> GroundednessResult:
    if finish_reason != "stop":
        return GroundednessResult(
            grounded=False,
            reason=(
                "Judge did not complete normally: "
                f"finish_reason={finish_reason!r}"
            ),
            judge_error=True,
            finish_reason=finish_reason,
            completion_tokens=completion_tokens,
        )

    if not content:
        return GroundednessResult(
            grounded=False,
            reason="Judge returned no content.",
            judge_error=True,
            finish_reason=finish_reason,
            completion_tokens=completion_tokens,
        )

    try:
        data = json.loads(content)
        result = GroundednessResult.model_validate(data)

        return result.model_copy(
            update={
                "finish_reason": finish_reason,
                "completion_tokens": completion_tokens,
            }
        )

    except (json.JSONDecodeError, ValueError):
        return GroundednessResult(
            grounded=False,
            reason="Judge returned invalid structured output.",
            judge_error=True,
            finish_reason=finish_reason,
            completion_tokens=completion_tokens,
        )


def assess_groundedness(
    claim: str,
    evidence: str,
) -> GroundednessResult:
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        max_tokens=800,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": (
                    f"EVIDENCE:\n{evidence}\n\n"
                    f"CLAIM:\n{claim}"
                ),
            },
        ],
    )

    choice = response.choices[0]
    content = choice.message.content or ""

    content = re.sub(
        r"<think>.*?</think>",
        "",
        content,
        flags=re.DOTALL,
    ).strip()
    usage = getattr(response, "usage", None)
    completion_tokens = (
      getattr(usage, "completion_tokens", None)
      if usage is not None
      else None
    )

    return parse_groundedness_response(
      content,
      choice.finish_reason,
      completion_tokens,
)
