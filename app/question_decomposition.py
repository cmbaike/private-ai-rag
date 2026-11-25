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


class DecompositionResult(BaseModel):
    requirements: list[str]


SYSTEM_PROMPT = """
You decompose a user's question into the minimum set of material
requirements that must ALL be supported by evidence for the original
question to be fully answered.

Your task is decomposition only. Do not answer the question.

Rules:

1. Preserve exactly what the user is asking for.
   Do not add related questions, background information, definitions,
   purposes, causes, examples, or implications unless explicitly requested.

2. Prefer ONE requirement.
   Split only when the original question explicitly requests multiple
   independently answerable facts, actions, or outputs.

3. A question asking for one value, fact, identity, date, version,
   quantity, region, duration, policy decision, or yes/no determination
   MUST normally produce one requirement.

4. Do not split conditions away from the thing they qualify.
   Example:
   "May AI restart production without human approval?"
   is ONE requirement about whether AI may restart production
   without human approval.

5. For explicit compound questions, create one requirement per requested
   component.
   Example:
   "Which region hosts production and which Kubernetes version does it run?"
   has TWO requirements:
   - the region hosting production
   - the Kubernetes version running in production

6. For procedures, split only independently requested operational goals.
   Example:
   "How do I check filesystem capacity and inode exhaustion?"
   has TWO requirements:
   - how to check filesystem capacity
   - how to check inode exhaustion

7. Preserve the subject, environment, scope, and qualifiers in EVERY
   requirement where needed.
   Do not change "production" to another environment.
   Do not change the grammatical subject.

8. Never substitute a related concept for the requested concept.
   RPO is not backup frequency.
   contractual SLA is not an internal target.
   production is not staging or disaster recovery.

9. Do not use outside technical knowledge.

10. Each requirement must be independently checkable against evidence.

11. Return the MINIMUM number of requirements needed to represent the
    original question faithfully.

Return JSON only:
{"requirements": ["...", "..."]}
"""


def decompose_question(question: str) -> DecompositionResult:
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        max_tokens=500,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"/no_think\nQuestion:\n{question}",
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

    result = DecompositionResult.model_validate(data)

    if not result.requirements:
        raise ValueError("Question decomposition returned no requirements")

    return result
