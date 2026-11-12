"""
Governing Law Risk Analyzer
===========================
Evaluates choice of substantive law, exclusive venue, dispute resolution mechanisms, and fee-shifting.
"""

from typing import Dict, Any, Optional, List
import re

from ml.risk_assessment.models import RiskLevel, AssessmentStatus, AssessmentFinding, ClauseExplanation
from ml.risk_assessment.rules.base import BaseClauseRuleAnalyzer


class GoverningLawAnalyzer(BaseClauseRuleAnalyzer):
    """Analyzes choice of law and dispute resolution provisions."""

    def __init__(self):
        super().__init__("governing_law")

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

        # Check for placeholder / blank jurisdiction
        has_placeholder = bool(re.search(r"\[[^\]]*(?:state|country|jurisdiction|insert)[^\]]*\]|__{2,}", text_lower))

        # Check for standard recognized commercial jurisdictions
        standard_commercial_jurisdictions = [
            "delaware", "new york", "california", "texas", "illinois",
            "england", "wales", "united kingdom", "massachusetts"
        ]
        has_standard_jurisdiction = any(j in text_lower for j in standard_commercial_jurisdictions)

        # Foreign / Offshore jurisdiction flag
        foreign_offshore_jurisdictions = [
            "cayman islands", "bermuda", "british virgin islands", "bvi", "cyprus", "panama"
        ]
        has_offshore_jurisdiction = any(j in text_lower for j in foreign_offshore_jurisdictions)

        # Exclusive venue / foreign court requirement
        has_exclusive_venue = bool(re.search(
            r"\b(?:exclusive\s+(?:jurisdiction|venue)|solely\s+in\s+the\s+courts\s+of)\b",
            text_lower
        ))

        # Arbitration and waivers
        has_arbitration = bool(re.search(r"\b(?:arbitrat\w+|aaa|icc|jams)\b", text_lower))
        has_jury_waiver = bool(re.search(r"\b(?:waive\w*\s+(?:any\s+)?right\s+to\s+(?:a\s+)?jury\s+trial|waiver\s+of\s+jury)\b", text_lower))
        has_class_action_waiver = bool(re.search(r"\b(?:class\s+action\s+waiver|waive\w*\s+class\s+action)\b", text_lower))

        # Unilateral attorney fees
        unilateral_fees = bool(re.search(
            r"\b(?:customer|client|licensee)\s+shall\s+(?:pay|reimburse)\s+(?:vendor|supplier|licensor|company)['’]?s?\s+(?:attorneys['’]?\s+fees|legal\s+costs)\b",
            text_lower
        ))
        mutual_fees = bool(re.search(
            r"\b(?:prevailing\s+party\s+shall\s+be\s+entitled|recover\s+reasonable\s+attorneys['’]?\s+fees)\b",
            text_lower
        ))

        # Collect evidence snippets
        jurisdiction_match = re.search(
            r"\b(?:governed\s+by|laws\s+of)\s+(?:the\s+state\s+of\s+)?([A-Za-z\s]+?)(?:law|laws|,|\.|\s+without|\s+and)",
            text, re.IGNORECASE
        )
        if jurisdiction_match:
            evidence.append(jurisdiction_match.group(0).strip())
        else:
            for j in standard_commercial_jurisdictions:
                if j in text_lower:
                    evidence.append(f"Designates {j.title()} jurisdiction.")
                    break
        if has_exclusive_venue:
            evidence.extend(self.extract_evidence_snippets(text, [r"\bexclusive\s+(?:jurisdiction|venue)\b"]))
        if unilateral_fees:
            evidence.extend(self.extract_evidence_snippets(text, [r"\b(?:attorneys['’]?\s+fees|legal\s+costs)\b"]))
        elif mutual_fees:
            evidence.extend(self.extract_evidence_snippets(text, [r"\bprevailing\s+party\b"]))

        # Decision Logic
        if has_placeholder:
            risk_level = RiskLevel.NEEDS_REVIEW
            risk_score = None
            risk_factors.append("unspecified_placeholder_jurisdiction")
            explanation = ClauseExplanation(
                summary="Governing law clause contains an uncompleted template placeholder.",
                why_it_matters="An unspecified jurisdiction leaves the entire contractual legal framework undefined.",
                supporting_evidence="Template brackets or blanks detected in the governing law text.",
                user_action="Specify an agreed neutral commercial jurisdiction before contract execution."
            )
        elif has_offshore_jurisdiction or unilateral_fees:
            risk_level = RiskLevel.HIGH
            risk_score = 7
            if has_offshore_jurisdiction:
                risk_factors.append("offshore_tax_haven_jurisdiction")
            if unilateral_fees:
                risk_factors.append("unilateral_attorney_fees_obligation")

            explanation = ClauseExplanation(
                summary="High-exposure governing law or unilateral legal fee liability.",
                why_it_matters="Offshore venues or one-sided fee-shifting create substantial expense and uncertainty in litigation.",
                supporting_evidence=" ".join(evidence) if evidence else "Governing law clause contains unilateral burdens or offshore venues.",
                user_action="Change venue to an established domestic jurisdiction and ensure attorney fee provisions are reciprocal."
            )
        elif has_standard_jurisdiction and not has_exclusive_venue and not unilateral_fees:
            risk_level = RiskLevel.LOW
            risk_score = 2
            risk_factors.append("standard_commercial_jurisdiction")
            if mutual_fees:
                risk_factors.append("mutual_prevailing_party_fees")

            matched_jurisdiction = next((j.title() for j in standard_commercial_jurisdictions if j in text_lower), "commercial")
            explanation = ClauseExplanation(
                summary=f"Standard commercial governing law ({matched_jurisdiction}) in an established, well-understood jurisdiction.",
                why_it_matters="Provides predictable commercial legal interpretation without onerous procedural barriers.",
                supporting_evidence=" ".join(evidence) if evidence else f"Recognized {matched_jurisdiction} jurisdiction.",
                user_action="Standard commercial provision; align internal legal counsel on designated state law."
            )
        else:
            # Medium risk: either non-standard domestic state, mandatory arbitration, or exclusive venue
            risk_level = RiskLevel.MEDIUM
            risk_score = 4
            if has_exclusive_venue:
                risk_factors.append("exclusive_foreign_or_distant_venue")
            if has_arbitration:
                risk_factors.append("mandatory_arbitration_clause")
            if has_jury_waiver:
                risk_factors.append("jury_trial_waiver")

            explanation = ClauseExplanation(
                summary="Governing law clause includes exclusive distant venue or mandatory arbitration.",
                why_it_matters="May require traveling to a distant forum to resolve disputes or waives court trial rights.",
                supporting_evidence=" ".join(evidence) if evidence else "Clause designates specific venue or dispute procedures.",
                user_action="Confirm that the designated venue is acceptable and not prohibitively distant from your operations."
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
