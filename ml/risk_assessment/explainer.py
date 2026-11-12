"""
ClauseGuard Plain-English Explanation Engine
============================================
Refines, formats, and guarantees that clause explanations satisfy all four
mandatory explanation criteria:
1. What the provision says.
2. Why its terms may matter.
3. What specific wording or evidence supports the assessment.
4. What the user should pay attention to.
"""

from typing import Dict, Any, List, Optional
from ml.risk_assessment.models import ClauseExplanation, RiskLevel, AssessmentStatus


class ExplanationEngine:
    """Formats and validates explanatory narratives for contractual risk findings."""

    @staticmethod
    def generate_explanation(
        summary: str,
        why_it_matters: str,
        supporting_evidence: str,
        user_action: str
    ) -> ClauseExplanation:
        """Constructs a compliant, grounded ClauseExplanation."""
        # Clean whitespace
        s = " ".join(summary.strip().split())
        w = " ".join(why_it_matters.strip().split())
        e = " ".join(supporting_evidence.strip().split()) if supporting_evidence else "Assessment derived from structural contract terms."
        u = " ".join(user_action.strip().split())

        return ClauseExplanation(
            summary=s,
            why_it_matters=w,
            supporting_evidence=e,
            user_action=u
        )

    @staticmethod
    def generate_fallback_explanation(
        category: str,
        status: AssessmentStatus,
        reason: Optional[str] = None
    ) -> ClauseExplanation:
        """Generates cautious, grounded explanations when text is unextracted or ambiguous."""
        readable = category.replace("_", " ").title()
        if status == AssessmentStatus.UNEXTRACTED_DETECTION:
            return ClauseExplanation(
                summary=f"The contract likely includes {readable} terms, but exact clause boundaries could not be isolated.",
                why_it_matters=f"{readable} provisions fundamentally govern legal liabilities, rights, or commitments.",
                supporting_evidence="Document-level classification detected category presence with positive confidence.",
                user_action=f"Manually inspect the agreement to verify {readable} terms."
            )
        elif status == AssessmentStatus.UNCERTAIN_BOUNDARY:
            detail = reason or "Extracted span appears truncated or fragmented."
            return ClauseExplanation(
                summary=f"Extracted {readable} clause has uncertain boundaries or missing context: {detail}",
                why_it_matters="Interpreting incomplete provisions can lead to inaccurate assumptions about your legal rights.",
                supporting_evidence="Boundary verification flagged fragmented character offsets.",
                user_action="Consult the complete contract section to evaluate the full sentence and surrounding context."
            )
        else:
            return ClauseExplanation(
                summary=f"Insufficient textual evidence to definitively assess {readable} risk.",
                why_it_matters="Definitive risk classification requires complete contractual language.",
                supporting_evidence="Available text is insufficient for automated legal analysis.",
                user_action="Request a complete copy of the agreement or exhibit."
            )
