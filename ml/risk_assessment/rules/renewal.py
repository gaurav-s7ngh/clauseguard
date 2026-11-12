"""
Renewal Term Risk Analyzer
==========================
Evaluates automatic renewal (evergreen), cancellation notice windows, renewal duration, and price escalation.
"""

from typing import Dict, Any, Optional, List
import re

from ml.risk_assessment.models import RiskLevel, AssessmentStatus, AssessmentFinding, ClauseExplanation
from ml.risk_assessment.rules.base import BaseClauseRuleAnalyzer


class RenewalTermAnalyzer(BaseClauseRuleAnalyzer):
    """Analyzes renewal mechanics and evergreen commitments."""

    def __init__(self):
        super().__init__("renewal_term")

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

        is_auto_renew = bool(re.search(
            r"\b(?:automatic(?:ally)?\s+renew\w*|renewal\s+shall\s+be\s+automatic|evergreen)\b",
            text_lower
        ))
        is_manual_renew = bool(re.search(
            r"\b(?:mutual\s+(?:written\s+)?agreement|only\s+upon\s+(?:mutual\s+)?written|upon\s+written\s+consent\s+of\s+both)\b",
            text_lower
        ))

        notice_days = self.extract_notice_days(text)

        # Multi-year renewal lock-in
        multi_year_lock = bool(re.search(
            r"\b(?:two|three|four|five|2|3|4|5)\s*(?:\([2-5]\)\s*)?years?\b",
            text_lower
        )) and is_auto_renew

        # Unilateral price increase on renewal
        price_escalation = bool(re.search(
            r"\b(?:then[\-\s]current\s+rates|increase\s+(?:fees|prices)|price\s+increase)\b",
            text_lower
        ))

        # Collect evidence
        if is_auto_renew:
            evidence.extend(self.extract_evidence_snippets(text, [r"\bautomatic\w*\s+renew\w*\b"]))
        if notice_days is not None:
            evidence.extend(self.extract_evidence_snippets(text, [r"\b(\d+)\s*(?:calendar|business)?\s*days?\b"]))
        if price_escalation:
            evidence.extend(self.extract_evidence_snippets(text, [r"\b(?:then[\-\s]current|price\s+increase)\b"]))

        # Decision Logic
        if is_manual_renew or (not is_auto_renew and "renew" in text_lower):
            risk_level = RiskLevel.LOW
            risk_score = 1
            risk_factors.append("mutual_written_renewal_only")
            explanation = ClauseExplanation(
                summary="Renewal requires affirmative mutual written consent of both parties.",
                why_it_matters="Prevents unintended contract extensions or surprise recurring billing; contract expires unless actively extended.",
                supporting_evidence=" ".join(evidence) if evidence else "Renewal conditional on mutual written agreement.",
                user_action="Set a calendar reminder 60 days before contract expiration if you intend to extend service."
            )
        elif is_auto_renew and (multi_year_lock or (notice_days is not None and notice_days >= 60) or price_escalation):
            risk_level = RiskLevel.HIGH
            risk_score = 8 if multi_year_lock else 7
            if multi_year_lock:
                risk_factors.append("multi_year_automatic_renewal_lockin")
            if notice_days is not None and notice_days >= 60:
                risk_factors.append(f"long_cancellation_window_{notice_days}_days")
            if price_escalation:
                risk_factors.append("unilateral_price_escalation_on_renewal")

            explanation = ClauseExplanation(
                summary="Aggressive auto-renewal provision with extended notice windows or multi-year financial lock-in.",
                why_it_matters="Missing the cancellation deadline will lock the company into an extended commitment or higher recurring fees.",
                supporting_evidence=" ".join(evidence) if evidence else "Automatic renewal with restrictive opt-out timeline.",
                user_action="Establish an immediate calendar alert well ahead of the non-renewal window, or negotiate 30 days notice."
            )
        elif is_auto_renew:
            # Standard auto-renewal: 1 year, 30 days notice
            risk_level = RiskLevel.MEDIUM
            risk_score = 4
            risk_factors.append("automatic_renewal_evergreen")
            if notice_days is not None:
                risk_factors.append(f"{notice_days}_days_cancellation_notice")

            explanation = ClauseExplanation(
                summary=f"Standard auto-renewal clause extending the term unless cancelled {notice_days or 30} days in advance.",
                why_it_matters="Ensures service continuity but requires proactive monitoring of renewal dates to avoid unwanted extensions.",
                supporting_evidence=" ".join(evidence) if evidence else "Term automatically renews unless notice of non-renewal is provided.",
                user_action=f"Track the contract renewal date and submit written non-renewal notice at least {notice_days or 30} days prior."
            )
        else:
            risk_level = RiskLevel.NEEDS_REVIEW
            risk_score = None
            risk_factors.append("renewal_mechanics_unclear")
            explanation = ClauseExplanation(
                summary="Renewal terms are referenced but exact duration or notice mechanics are unspecified.",
                why_it_matters="Unclear renewal terms can create confusion over when the contract formally terminates.",
                supporting_evidence=" ".join(evidence) if evidence else "Clause text references renewal without explicit mechanics.",
                user_action="Clarify whether renewal is automatic or requires mutual written agreement."
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
