from pydantic import BaseModel

from app.question_decomposition import decompose_question
from app.sufficiency import assess_sufficiency, SufficiencyResult
from app.sufficiency_consistency import apply_consistency_guard


class RequirementAssessment(BaseModel):
    requirement: str
    raw_sufficient: bool
    guarded_sufficient: bool
    reason: str
    guard_triggered: bool


class AtomicSufficiencyResult(BaseModel):
    sufficient: bool
    requirements: list[RequirementAssessment]


def assess_atomic_sufficiency(
    question: str,
    sources: list[dict],
) -> AtomicSufficiencyResult:
    decomposition = decompose_question(question)

    assessments = []

    for requirement in decomposition.requirements:
        raw: SufficiencyResult = assess_sufficiency(
            requirement,
            sources,
        )

        guarded = apply_consistency_guard(raw)

        assessments.append(
            RequirementAssessment(
                requirement=requirement,
                raw_sufficient=raw.sufficient,
                guarded_sufficient=guarded.sufficient,
                reason=raw.reason,
                guard_triggered=(
                    raw.sufficient != guarded.sufficient
                ),
            )
        )

    # Fail closed:
    # every material requirement must be supported.
    sufficient = (
        bool(assessments)
        and all(
            assessment.guarded_sufficient
            for assessment in assessments
        )
    )

    return AtomicSufficiencyResult(
        sufficient=sufficient,
        requirements=assessments,
    )
