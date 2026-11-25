import json
import os
import re

from openai import OpenAI
from pydantic import BaseModel, ValidationError


BASE_URL = os.getenv(
    "CLAIM_EXTRACTION_BASE_URL",
    "http://localhost:11434/v1",
)
API_KEY = os.getenv("CLAIM_EXTRACTION_API_KEY", "ollama")
MODEL = os.getenv("CLAIM_EXTRACTION_MODEL", "gpt-oss:20b")

client = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
)

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


class ExtractedClaim(BaseModel):
    claim: str
    citations: list[str]


class ClaimExtractionResult(BaseModel):
    claims: list[ExtractedClaim]
    extraction_error: bool = False
    reason: str | None = None
    finish_reason: str | None = None
    completion_tokens: int | None = None


SYSTEM_PROMPT = """
You extract material factual claims from an AI assistant answer.

Return each material factual assertion as a separate atomic claim and
associate it with the explicit SOURCE citations that support that claim
according to the answer.

Rules:

1. Extract factual assertions about the system, incident, configuration,
   procedure, policy, state, cause, consequence, timing, numeric value,
   permission, prohibition, or operational behavior.

2. Split compound assertions into separate claims when the parts could
   independently be supported or unsupported.

3. Preserve the meaning and scope of the answer. Do not strengthen,
   weaken, correct, or add information.

4. Do not decide whether a claim is true or grounded. Extraction only.

5. For each claim, include only SOURCE labels explicitly attached to or
   clearly used by the answer for that claim.

6. Preserve SOURCE labels exactly as written inside the citation.

7. If a material factual claim has no citation, return it with an empty
   citations list.

8. Do not treat the citation label itself as a factual claim.

9. Do not extract purely conversational framing such as
   "Based on the provided context".

10. Do not merge separate factual propositions merely because they
    occur in the same sentence.

Return JSON only in this form:

{
  "claims": [
    {
      "claim": "atomic factual claim",
      "citations": ["exact source label"]
    }
  ]
}

If there are no material factual claims, return:

{"claims": []}
""".strip()


def extract_claims(answer: str) -> ClaimExtractionResult:
    try:
        response = client.chat.completions.create(
            model=MODEL,
            temperature=0,
            max_tokens=800,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": answer},
            ],
        )
    except Exception as exc:
        return ClaimExtractionResult(
            claims=[],
            extraction_error=True,
            reason=f"request failed: {exc}",
        )

    choice = response.choices[0]
    finish_reason = getattr(choice, "finish_reason", None)

    usage = getattr(response, "usage", None)
    completion_tokens = (
        getattr(usage, "completion_tokens", None)
        if usage is not None
        else None
    )

    if finish_reason != "stop":
        return ClaimExtractionResult(
            claims=[],
            extraction_error=True,
            reason=f"non-stop finish_reason: {finish_reason}",
            finish_reason=finish_reason,
            completion_tokens=completion_tokens,
        )

    content = choice.message.content

    if not content:
        return ClaimExtractionResult(
            claims=[],
            extraction_error=True,
            reason="empty model response",
            finish_reason=finish_reason,
            completion_tokens=completion_tokens,
        )

    content = THINK_RE.sub("", content).strip()

    try:
        payload = json.loads(content)
        claims = [
            ExtractedClaim.model_validate(item)
            for item in payload["claims"]
        ]
    except (
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValidationError,
    ) as exc:
        return ClaimExtractionResult(
            claims=[],
            extraction_error=True,
            reason=f"invalid extraction response: {exc}",
            finish_reason=finish_reason,
            completion_tokens=completion_tokens,
        )

    return ClaimExtractionResult(
        claims=claims,
        extraction_error=False,
        finish_reason=finish_reason,
        completion_tokens=completion_tokens,
    )
