"""
Termination for Convenience Risk Analyzer
=========================================
Evaluates termination without cause rights, notice periods, mutuality, and penalty fees.
"""

from typing import Dict, Any, Optional, List
import re

from ml.risk_assessment.models import RiskLevel, AssessmentStatus, AssessmentFinding, ClauseExplanation
from ml.risk_assessment.rules.base import BaseClauseRuleAnalyzer


class TerminationAnalyzer(BaseClauseRuleAnalyzer):
    """Analyzes termination for convenience provisions."""

    def __init__(self):
        super().__init__("termination_for_convenience")

    def analyze(
        self,
        clause_dict: Dict[str, Any],
        document_context: Optional[str] = None
    ) -> AssessmentFinding:
        early_exit = self.check_minimum_evidence(clause_dict)
        if early_exit is not None:
            return early_exit

        text = clause_dict["clause_text"]
        doc_id = clause_dict.get("document_id", "unknown_doc")
        conf = float(clause_dict.get("confidence", 0.0))
        text_lower = text.lower()

        risk_factors: List[str] = []
        evidence: List[str] = []

        is_mutual = self.is_mutual(text)
        notice_days = self.extract_notice_days(text)

        # Unilateral patterns
        unilateral_other = bool(re.search(
            r"\b(?:vendor|supplier|licensor|company)\s+may\s+terminate\b",
            text_lower
        )) and not is_mutual

        # Penalties and fee forfeiture
        has_penalty_fee = bool(re.search(
            r"\b(?:termination\s+fee|early\s+termination\s+penalty|liquidated\s+damages|forfeit\s+(?:all\s+)?prepaid|accelerate\s+all\s+(?:unpaid\s+)?fees)\b",
            text_lower
        ))
        has_refund = bool(re.search(
            r"\b(?:pro\s*rata\s+refund|refund\s+of\s+unearned|reimburse\s+prepaid)\b",
            text_lower
        ))

        # Collect evidence snippets
        if notice_days is not None:
            evidence.extend(self.extract_evidence_snippets(text, [r"(\d+|immediate\w*|without\s+notice)\s*(?:calendar|business)?\s*days?"]))
        if has_penalty_fee:
            evidence.extend(self.extract_evidence_snippets(text, [r"\b(?:termination\s+fee|early\s+termination|liquidated\s+damages)\b"]))
        if is_mutual:
            evidence.extend(self.extract_evidence_snippets(text, [r"\b(?:either|both|each)\s+party\b"]))
        elif unilateral_other:
            evidence.extend(self.extract_evidence_snippets(text, [r"\b(?:vendor|supplier|licensor|company)\s+may\s+terminate\b"]))

        # Decision Logic
        if notice_days is None and not any(k in text_lower for k in ["notice", "immediate", "terminate"]):
            risk_level = RiskLevel.NEEDS_REVIEW
            risk_score = None
            risk_factors.append("notice_period_undetermined")
            explanation = ClauseExplanation(
                summary="Termination clause does not clearly specify the required notice timeline.",
                why_it_matters="Without defined notice periods, termination mechanics remain legally ambiguous.",
                supporting_evidence="No explicit day count or notice mechanism identified in the extracted text.",
                user_action="Clarify the required written notice period (standard is 30 to 60 days)."
            )
        elif (unilateral_other and (notice_days is not None and notice_days < 14)) or has_penalty_fee:
            risk_level = RiskLevel.HIGH
            risk_score = 9 if has_penalty_fee else 8
            if has_penalty_fee:
                risk_factors.append("early_termination_penalty")
            if unilateral_other:
                risk_factors.append("unilateral_termination_right")
            if notice_days is not None and notice_days < 14:
                risk_factors.append("short_notice_period")

            explanation = ClauseExplanation(
                summary="Highly restrictive or asymmetrical termination right with financial penalties or minimal notice.",
                why_it_matters="Allows abrupt cancellation of services or imposes punitive fees if the user needs to exit the contract.",
                supporting_evidence=" ".join(evidence) if evidence else "Terms permit termination on short notice or penalize early exit.",
                user_action="Negotiate at least 30 days prior written notice and strike any early termination penalty fees."
            )
        elif is_mutual and notice_days is not None and notice_days >= 30:
            risk_level = RiskLevel.LOW
            risk_score = 2
            risk_factors.append("mutual_convenience_termination")
            risk_factors.append(f"{notice_days}_days_notice")
            if has_refund:
                risk_factors.append("prorata_refund_guaranteed")

            explanation = ClauseExplanation(
                summary=f"Balanced mutual termination right upon {notice_days} days prior written notice.",
                why_it_matters="Provides an orderly, symmetrical off-ramp for both parties without requiring a breach of contract.",
                supporting_evidence=" ".join(evidence) if evidence else f"Mutual termination permitted upon {notice_days} days notice.",
                user_action="Confirm that operational dependencies can be transitioned within this notice window."
            )
        else:
            # Moderate risk: unilateral with 30d notice, or mutual with short 15d notice
            risk_level = RiskLevel.MEDIUM
            risk_score = 5
            if unilateral_other:
                risk_factors.append("unilateral_termination_right")
            if notice_days is not None and notice_days < 30:
                risk_factors.append("notice_under_30_days")
            else:
                risk_factors.append("asymmetrical_termination_procedure")

            explanation = ClauseExplanation(
                summary="Moderate termination provision with unilateral authority or compressed notice windows.",
                why_it_matters="May create operational uncertainty if notice is inadequate or rights are not strictly reciprocal.",
                supporting_evidence=" ".join(evidence) if evidence else "Termination allowed with asymmetrical rights or compressed notice.",
                user_action="Request mutual reciprocal termination rights with at least 30 to 60 days advance notice."
            )

        return AssessmentFinding(
            document_id=doc_id,
            category=self.category_name,
            clause_text=text,
            start_char=clause_dict.get("start_char"),
            end_char=clause_dict.get("end_char"),
            confidence=conf,
            risk_level=risk_level,
            risk_score=risk_score,
            explanation=explanation,
            evidence=evidence,
            risk_factors=risk_factors,
            extraction_status=clause_dict.get("extraction_status", "extracted"),
            assessment_status=AssessmentStatus.ASSESSED if risk_level != RiskLevel.NEEDS_REVIEW else AssessmentStatus.UNCERTAIN_BOUNDARY,
            review_required=(risk_level in [RiskLevel.HIGH, RiskLevel.NEEDS_REVIEW])
        )
