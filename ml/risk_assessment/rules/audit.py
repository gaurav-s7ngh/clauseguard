"""
Audit Rights Risk Analyzer
==========================
Evaluates inspection rights, advance notice, frequency, cost allocation, and confidentiality safeguards.
"""

from typing import Dict, Any, Optional, List
import re

from ml.risk_assessment.models import RiskLevel, AssessmentStatus, AssessmentFinding, ClauseExplanation
from ml.risk_assessment.rules.base import BaseClauseRuleAnalyzer


class AuditRightsAnalyzer(BaseClauseRuleAnalyzer):
    """Analyzes audit and inspection provisions."""

    def __init__(self):
        super().__init__("audit_rights")

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

        notice_days = self.extract_notice_days(text)

        # Unannounced or immediate audit
        unannounced = bool(re.search(
            r"\b(?:without\s+(?:prior\s+)?notice|at\s+any\s+time|immediate\s+access|unannounced)\b",
            text_lower
        ))

        # Frequency limitations
        once_per_year = bool(re.search(
            r"\b(?:no\s+more\s+(?:often\s+)?than\s+once|not\s+more\s+than\s+once|once\s+per\s+(?:calendar\s+)?year|once\s+in\s+any\s+twelve)\b",
            text_lower
        ))
        unlimited_frequency = bool(re.search(
            r"\b(?:from\s+time\s+to\s+time|as\s+often\s+as|whenever\s+(?:requested|deemed\s+necessary))\b",
            text_lower
        ))

        # Business hours & non-disruption
        during_business_hours = bool(re.search(
            r"\b(?:normal|regular)\s+business\s+hours\b",
            text_lower
        ))
        non_disruption = bool(re.search(
            r"\b(?:not\s+(?:unreasonably\s+)?disrupt|minimize\s+disruption)\b",
            text_lower
        ))

        # Cost allocation
        audited_pays_always = bool(re.search(
            r"\b(?:customer|audited\s+party|licensee)\s+shall\s+(?:bear|pay)\s+all\s+(?:costs|expenses)\s+of\s+(?:the\s+|such\s+)?audit\b",
            text_lower
        )) and not bool(re.search(r"\b(?:if|unless|in\s+the\s+event)\s+(?:an?\s+)?underpayment\b", text_lower))

        standard_cost_contingency = bool(re.search(
            r"\b(?:exceeds\s+5%|underpayment\s+of\s+more\s+than\s+(?:five|5)%|material\s+deficiency)\b",
            text_lower
        ))

        # Confidentiality requirement for auditor
        confidentiality_safeguard = bool(re.search(
            r"\b(?:confidentiality\s+agreement|independent\s+(?:certified\s+)?auditor|subject\s+to\s+confidentiality)\b",
            text_lower
        ))

        # Collect evidence snippets
        if unannounced:
            evidence.extend(self.extract_evidence_snippets(text, [r"\b(?:without\s+notice|immediate\s+access|at\s+any\s+time)\b"]))
        if notice_days is not None:
            evidence.extend(self.extract_evidence_snippets(text, [r"\b(\d+)\s*(?:calendar|business)?\s*days?\b"]))
        if once_per_year:
            evidence.extend(self.extract_evidence_snippets(text, [r"\bonce\s+per\s+year\b", r"\bnot\s+more\s+than\s+once\b"]))
        if audited_pays_always:
            evidence.extend(self.extract_evidence_snippets(text, [r"\bshall\s+bear\s+all\s+costs\b"]))

        # Decision Logic
        if unannounced or audited_pays_always or (unlimited_frequency and notice_days is not None and notice_days < 5):
            risk_level = RiskLevel.HIGH
            risk_score = 8
            if unannounced:
                risk_factors.append("unannounced_immediate_audits")
            if audited_pays_always:
                risk_factors.append("audited_party_pays_all_audit_costs")
            if unlimited_frequency:
                risk_factors.append("unlimited_audit_frequency")

            explanation = ClauseExplanation(
                summary="Intrusive audit right with immediate access or unfair cost burdens.",
                why_it_matters="Can disrupt day-to-day operations and impose significant unexpected audit fees without reasonable notice.",
                supporting_evidence=" ".join(evidence) if evidence else "Audit terms permit immediate access or shift audit fees unconditionally.",
                user_action="Require at least 15-30 business days notice, limit to once per year, and condition fee-shifting on an underpayment > 5%."
            )
        elif once_per_year and during_business_hours and (notice_days is not None and notice_days >= 10):
            risk_level = RiskLevel.LOW
            risk_score = 2
            risk_factors.append("limited_to_annual_audit")
            risk_factors.append(f"{notice_days}_days_prior_notice")
            if during_business_hours:
                risk_factors.append("regular_business_hours_only")
            if confidentiality_safeguard:
                risk_factors.append("auditor_confidentiality_bound")

            explanation = ClauseExplanation(
                summary="Standard, well-hedged audit clause with customary operational safeguards.",
                why_it_matters="Allows legitimate contract verification while protecting business operations and confidentiality.",
                supporting_evidence=" ".join(evidence) if evidence else f"Annual audit with {notice_days} days advance notice during business hours.",
                user_action="Ensure records are maintained in an orderly format to facilitate smooth annual verification."
            )
        else:
            # Medium risk: missing notice days, or unlimited frequency with standard notice
            risk_level = RiskLevel.MEDIUM
            risk_score = 5
            if notice_days is not None and notice_days < 10:
                risk_factors.append("short_audit_notice_under_10_days")
            if not once_per_year:
                risk_factors.append("unrestricted_audit_frequency")
            if not confidentiality_safeguard:
                risk_factors.append("missing_auditor_nda_requirement")

            explanation = ClauseExplanation(
                summary="Audit provision lacking complete operational or frequency safeguards.",
                why_it_matters="Could permit multiple inspections per year or provide inadequate preparation time.",
                supporting_evidence=" ".join(evidence) if evidence else "Audit clause lacks clear annual frequency caps or advance notice minimums.",
                user_action="Limit audit frequency to once per 12 months and mandate at least 15 business days advance notice."
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
