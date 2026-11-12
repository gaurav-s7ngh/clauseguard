"""
Liability Cap Risk Analyzer
===========================
Evaluates monetary liability limits, mutuality, exclusions, and benchmarks.
"""

from typing import Dict, Any, Optional, List
import re

from ml.risk_assessment.models import RiskLevel, AssessmentStatus, AssessmentFinding, ClauseExplanation
from ml.risk_assessment.rules.base import BaseClauseRuleAnalyzer


class LiabilityCapAnalyzer(BaseClauseRuleAnalyzer):
    """Analyzes limitation of liability provisions."""

    def __init__(self):
        super().__init__("liability_cap")

    def analyze(
        self,
        clause_dict: Dict[str, Any],
        document_context: Optional[str] = None
    ) -> AssessmentFinding:
        # Check minimum evidence
        early_exit = self.check_minimum_evidence(clause_dict)
        if early_exit is not None:
            return early_exit

        text = clause_dict["clause_text"]
        doc_id = clause_dict.get("document_id", "unknown_doc")
        conf = float(clause_dict.get("confidence", 0.0))
        text_lower = text.lower()

        risk_factors: List[str] = []
        evidence: List[str] = []

        # 1. Mutuality check
        is_mutual = self.is_mutual(text)

        # Check if cap applies strictly to supplier / vendor / licensor
        vendor_one_sided = bool(re.search(
            r"\b(?:vendor|supplier|licensor|contractor|service\s+provider|company)['’]?s?\s+(?:total|aggregate|maximum)?\s*liability\s+(?:shall|will|is)\b",
            text_lower
        ))
        customer_one_sided = bool(re.search(
            r"\b(?:customer|client|licensee)['’]?s?\s+(?:total|aggregate|maximum)?\s*liability\s+(?:shall|will|is)\b",
            text_lower
        ))

        # 2. Benchmark evaluation
        has_12_month_fees = bool(re.search(r"\b(?:twelve|12)(?:\s*\(\s*12\s*\))?\s+months?\b", text_lower) and "fees" in text_lower)
        has_nominal_cap = bool(re.search(r"\$(?:[1-9]\d{0,2}|0)(?:\.00)?\b|\b(?:one\s+hundred|fifty)\s+dollars\b|\bfees\s+paid\s+in\s+the\s+(?:prior|preceding)\s+month\b", text_lower))
        has_substantial_fixed_sum = bool(re.search(r"\$(?:[1-9]\d{5,}|[1-9]\d{2,},\d{3})\b", text_lower))  # >= $100,000

        # Disclaimer of all liability (zero or disclaimer of direct damages)
        complete_disclaimer = bool(re.search(r"\b(?:in\s+no\s+event\s+shall|no\s+liability\s+for|disclaims\s+all\s+liability|shall\s+not\s+be\s+liable\s+for\s+any\s+(?:direct|actual|compensatory)\s+damages)\b", text_lower))

        # 3. Carve-out evaluation
        has_confidentiality_carveout = bool(re.search(r"\b(?:confidentiality|breach\s+of\s+section\s+\d+)\b", text_lower))
        has_indemnity_carveout = bool(re.search(r"\b(?:indemnif(?:y|ication)|ip\s+infringement)\b", text_lower))
        has_gross_negligence = bool(re.search(r"\b(?:gross\s+negligence|willful\s+misconduct)\b", text_lower))
        has_carveouts = has_confidentiality_carveout or has_indemnity_carveout or has_gross_negligence

        # 4. Ambiguity / External reference check
        has_external_schedule = bool(re.search(r"\b(?:as\s+set\s+forth\s+in|specified\s+in|per)\s+(?:schedule|exhibit|order\s+form|sow)\b", text_lower))

        # Collect evidence snippets
        if is_mutual:
            evidence.extend(self.extract_evidence_snippets(text, [r"\b(?:either|both|neither)\s+party\b", r"\beach\s+party\b"]))
        elif vendor_one_sided:
            evidence.extend(self.extract_evidence_snippets(text, [r"\b(?:vendor|supplier|licensor)['’]?s?\s+liability\b"]))
        if has_12_month_fees:
            evidence.extend(self.extract_evidence_snippets(text, [r"\b(?:twelve|12)\s+months?\b"]))
        if has_carveouts:
            evidence.extend(self.extract_evidence_snippets(text, [r"\b(?:confidentiality|indemnif\w+|gross\s+negligence)\b"]))

        # Decision Logic
        if has_external_schedule and not (has_12_month_fees or has_substantial_fixed_sum or has_nominal_cap):
            risk_level = RiskLevel.NEEDS_REVIEW
            risk_score = None
            risk_factors.append("cap_amount_in_unattached_schedule")
            explanation = ClauseExplanation(
                summary="Liability cap relies on external schedules or order forms not contained in the extracted text.",
                why_it_matters="The exact monetary ceiling cannot be determined without reviewing the referenced schedule.",
                supporting_evidence="Clause references liability amounts specified in an external exhibit or order form.",
                user_action="Obtain and review the referenced order form to verify the exact liability limit."
            )
        elif complete_disclaimer or has_nominal_cap or (vendor_one_sided and not is_mutual and not has_carveouts):
            risk_level = RiskLevel.HIGH
            risk_score = 8 if not complete_disclaimer else 9
            if complete_disclaimer:
                risk_factors.append("complete_liability_disclaimer")
            if has_nominal_cap:
                risk_factors.append("nominal_liability_cap")
            if vendor_one_sided and not is_mutual:
                risk_factors.append("unilateral_vendor_protection")
            if not has_carveouts:
                risk_factors.append("missing_customary_carveouts")

            explanation = ClauseExplanation(
                summary="One-sided or severely restricted liability limitation that creates asymmetrical legal exposure.",
                why_it_matters="A low or unilateral cap prevents meaningful financial recovery if the vendor causes major harm or breaches the contract.",
                supporting_evidence=" ".join(evidence) if evidence else "Terms establish a one-sided cap without reciprocal protection.",
                user_action="Negotiate a mutual cap tied to at least 12 months fees with express carve-outs for confidentiality and IP indemnity."
            )
        elif is_mutual and (has_12_month_fees or has_substantial_fixed_sum):
            risk_level = RiskLevel.LOW
            risk_score = 2
            risk_factors.append("mutual_balanced_cap")
            if has_carveouts:
                risk_factors.append("customary_carveouts_present")

            explanation = ClauseExplanation(
                summary="Standard mutual liability cap tied to fees paid or a substantial fixed amount.",
                why_it_matters="Protects both parties symmetrically against catastrophic uncapped liability under normal operations.",
                supporting_evidence=" ".join(evidence) if evidence else "Mutual liability ceiling tied to standard commercial fee benchmarks.",
                user_action="Verify that insurance coverage matches or exceeds this aggregate limitation."
            )
        else:
            # Moderate risk: either one-sided with reasonable 12m fees, or mutual with vague/narrow terms
            risk_level = RiskLevel.MEDIUM
            risk_score = 5
            if not is_mutual:
                risk_factors.append("one_sided_cap")
            if not has_carveouts:
                risk_factors.append("no_confidentiality_or_indemnity_carveouts")

            explanation = ClauseExplanation(
                summary="Moderate liability limitation with partial protections or missing carve-outs.",
                why_it_matters="While an aggregate limit exists, lack of carve-outs or mutuality could leave specific claims uncompensated.",
                supporting_evidence=" ".join(evidence) if evidence else "Standard fee cap lacking comprehensive mutual protections.",
                user_action="Ensure confidentiality breaches and gross negligence are explicitly excluded from the limitation."
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
