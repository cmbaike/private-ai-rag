import json
import os
import re

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel


load_dotenv()

client = OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1"),
    api_key=os.getenv("OPENAI_API_KEY", "local"),
)

MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen3-4B")


class ValueSufficiencyResult(BaseModel):
    sufficient: bool
    reason: str


SYSTEM_PROMPT = """
You determine whether supplied evidence explicitly states the actual
value requested by a requirement.

This is a strict evidence-binding task.

Rules:

1. Return sufficient=true ONLY when the evidence explicitly binds the
   requested attribute to its actual value.

2. Mentioning the requested attribute is not enough.

3. Saying where the value is defined, stored, documented, configured,
   maintained, or can be retrieved is not the value.

4. A value belonging to a related but different attribute must not be
   substituted for the requested value.

Examples of invalid substitutions:
- incident acknowledgement time is not disaster recovery RTO
- backup frequency is not backup retention
- internal availability target is not contractual SLA
- staging region is not production region
- disaster recovery region is not primary production region
- kubectl client version is not production cluster version

5. Preserve scope and qualifiers exactly:
   production != staging
   production != disaster recovery
   contractual != internal

6. Do not infer, calculate, estimate, or use outside knowledge.

7. If the requested value must be looked up somewhere else, return false.

8. If evidence provides several values, return true only if the evidence
   explicitly associates the correct value with the requested attribute.

9. If uncertain, return false.

Return JSON only:
{"sufficient": true/false, "reason": "..."}
"""


def assess_value_sufficiency(
    requirement: str,
    sources: list[dict],
) -> ValueSufficiencyResult:
    evidence = "\n\n".join(
        source["content"]
        for source in sources
    )

    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        max_tokens=500,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "/no_think\n"
                    f"Required value:\n{requirement}\n\n"
                    f"Evidence:\n{evidence}"
                ),
            },
        ],
    )

    content = response.choices[0].message.content or ""

    content = re.sub(
        r"<think>.*?</think>",
        "",
        content,
        flags=re.DOTALL,
    ).strip()

    data = json.loads(content)

    return ValueSufficiencyResult.model_validate(data)
