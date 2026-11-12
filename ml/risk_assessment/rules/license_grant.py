"""
License Grant Risk Analyzer
===========================
Evaluates license exclusivity, scope, revocability, sublicensing, and operational restrictions.
"""

from typing import Dict, Any, Optional, List
import re

from ml.risk_assessment.models import RiskLevel, AssessmentStatus, AssessmentFinding, ClauseExplanation
from ml.risk_assessment.rules.base import BaseClauseRuleAnalyzer


class LicenseGrantAnalyzer(BaseClauseRuleAnalyzer):
    """Analyzes software, content, and patent license grant provisions."""

    def __init__(self):
        super().__init__("license_grant")

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

        # Exclusivity check
        is_exclusive = bool(re.search(r"\bexclusive\s+(?:license|right)\b", text_lower)) and not bool(re.search(r"\bnon[\-\s]exclusive\b", text_lower))
        is_non_exclusive = bool(re.search(r"\bnon[\-\s]exclusive\b", text_lower))

        # Revocability
        is_revocable = bool(re.search(r"\brevocable\s+(?:at\s+(?:any\s+time|will)|license)\b", text_lower)) and not bool(re.search(r"\birrevocable\b", text_lower))
        is_irrevocable = bool(re.search(r"\birrevocable\b", text_lower))

        # Sublicensing
        has_sublicensing = bool(re.search(r"\b(?:with\s+the\s+right\s+to\s+sublicense|sublicensable)\b", text_lower))
        no_sublicensing = bool(re.search(r"\bnon[\-\s]sublicensable|without\s+the\s+right\s+to\s+sublicense\b", text_lower))

        # Perpetual vs term-limited
        is_perpetual = bool(re.search(r"\bperpetual\b", text_lower))
        is_royalty_free = bool(re.search(r"\b(?:fully\s+paid[\-\s]up|royalty[\-\s]free)\b", text_lower))

        # Collect evidence
        if is_exclusive:
            evidence.extend(self.extract_evidence_snippets(text, [r"\bexclusive\s+(?:license|right)\b"]))
        elif is_non_exclusive:
            evidence.extend(self.extract_evidence_snippets(text, [r"\bnon[\-\s]exclusive\b"]))
        if is_revocable:
            evidence.extend(self.extract_evidence_snippets(text, [r"\brevocable\b"]))
        elif is_irrevocable:
            evidence.extend(self.extract_evidence_snippets(text, [r"\birrevocable\b"]))
        if is_perpetual:
            evidence.extend(self.extract_evidence_snippets(text, [r"\bperpetual\b"]))

        # Decision Logic
        if is_exclusive or is_revocable:
            risk_level = RiskLevel.HIGH
            risk_score = 7
            if is_exclusive:
                risk_factors.append("exclusive_license_grant")
            if is_revocable:
                risk_factors.append("at_will_revocable_license")

            explanation = ClauseExplanation(
                summary="Exclusive or arbitrarily revocable license grant creating operational vulnerability.",
                why_it_matters="An exclusive grant blocks you from licensing your IP elsewhere; a revocable license leaves operations exposed to sudden withdrawal.",
                supporting_evidence=" ".join(evidence) if evidence else "License terms contain exclusive encumbrances or at-will revocation rights.",
                user_action="Ensure software licenses are irrevocable during the paid term and commercial grants are strictly non-exclusive."
            )
        elif is_non_exclusive and (is_irrevocable or is_royalty_free or not is_revocable):
            risk_level = RiskLevel.LOW
            risk_score = 2
            risk_factors.append("non_exclusive_license")
            if is_irrevocable:
                risk_factors.append("irrevocable_during_term")
            if is_royalty_free:
                risk_factors.append("fully_paid_up_royalty_free")

            explanation = ClauseExplanation(
                summary="Standard non-exclusive commercial license grant for legitimate business use.",
                why_it_matters="Grants necessary rights to use the software or technology without imposing restrictive exclusivity.",
                supporting_evidence=" ".join(evidence) if evidence else "Non-exclusive commercial license grant.",
                user_action="Confirm that named user or volume tiers accommodate expected operational growth."
            )
        else:
            # Medium risk: ambiguous scope, rigid usage constraints
            risk_level = RiskLevel.MEDIUM
            risk_score = 4
            if no_sublicensing:
                risk_factors.append("no_sublicense_to_contractors_or_affiliates")
            risk_factors.append("standard_commercial_restrictions")

            explanation = ClauseExplanation(
                summary="Moderate license grant with standard operational restrictions.",
                why_it_matters="May restrict third-party contractors or subsidiaries from accessing the system on your behalf.",
                supporting_evidence=" ".join(evidence) if evidence else "Commercial license grant with specific usage limitations.",
                user_action="Verify that third-party consultants and affiliates are permitted to access the license under your account."
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
            assessment_status=AssessmentStatus.ASSESSED,
            review_required=(risk_level == RiskLevel.HIGH)
        )
