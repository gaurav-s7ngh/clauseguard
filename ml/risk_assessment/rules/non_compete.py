"""
Non-Compete Risk Analyzer
=========================
Evaluates restrictive covenants, duration, geographic breadth, and activity constraints.
Avoids unsupported legal enforceability conclusions while highlighting contractual exposure.
"""

from typing import Dict, Any, Optional, List
import re

from ml.risk_assessment.models import RiskLevel, AssessmentStatus, AssessmentFinding, ClauseExplanation
from ml.risk_assessment.rules.base import BaseClauseRuleAnalyzer


class NonCompeteAnalyzer(BaseClauseRuleAnalyzer):
    """Analyzes non-competition and non-solicitation restrictive covenants."""

    def __init__(self):
        super().__init__("non_compete")

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

        # Duration analysis
        multi_year = bool(re.search(
            r"\b(?:two|three|four|five|2|3|4|5)\s*(?:\([2-5]\)\s*)?years?\b",
            text_lower
        ))
        one_year = bool(re.search(
            r"\b(?:one|1)\s*(?:\(1\)\s*)?year\b|\b(?:twelve|12)\s+months?\b",
            text_lower
        ))
        six_months_or_less = bool(re.search(
            r"\b(?:six|6|three|3)\s*(?:\([36]\)\s*)?months?\b",
            text_lower
        ))
        perpetual_compete = bool(re.search(
            r"\b(?:perpetual|indefinite|at\s+all\s+times\s+hereafter)\b",
            text_lower
        ))

        # Geographic scope
        worldwide_scope = bool(re.search(
            r"\b(?:worldwide|anywhere\s+in\s+the\s+world|global|any\s+(?:geographic\s+)?market|any\s+territory\s+in\s+which)\b",
            text_lower
        ))

        # Non-solicit vs pure non-compete
        is_non_solicit_only = bool(re.search(
            r"\b(?:(?:not|shall\s+not|neither\s+party\s+shall)\s+solicit\s+(?:any\s+)?(?:employee|customer|client)|non[\-\s]solicitation|solicit\s+any\s+(?:employee|customer))\b",
            text_lower
        )) and not bool(re.search(r"\bnot\s+engage\s+in\s+any\s+(?:competing\s+)?business\b", text_lower))

        broad_activity_ban = bool(re.search(
            r"\b(?:any\s+business|directly\s+or\s+indirectly\s+(?:engage|compete|participate)|similar\s+to\s+or\s+competitive\s+with)\b",
            text_lower
        ))

        # Collect evidence
        if perpetual_compete:
            evidence.extend(self.extract_evidence_snippets(text, [r"\b(?:perpetual|indefinite|at\s+all\s+times)\b"]))
        elif multi_year:
            evidence.extend(self.extract_evidence_snippets(text, [r"\b(?:two|three|four|five|2|3|4|5)\s*years?\b"]))
        elif one_year:
            evidence.extend(self.extract_evidence_snippets(text, [r"\b(?:one|1)\s*year\b", r"\b12\s+months?\b"]))
        if worldwide_scope:
            evidence.extend(self.extract_evidence_snippets(text, [r"\b(?:worldwide|global|any\s+territory)\b"]))
        if broad_activity_ban:
            evidence.extend(self.extract_evidence_snippets(text, [r"\b(?:directly\s+or\s+indirectly|any\s+business)\b"]))

        # Decision Logic
        if perpetual_compete or (multi_year and worldwide_scope) or (broad_activity_ban and multi_year):
            risk_level = RiskLevel.HIGH
            risk_score = 9 if perpetual_compete else 8
            if perpetual_compete:
                risk_factors.append("perpetual_restrictive_covenant")
            if multi_year:
                risk_factors.append("multi_year_non_compete_duration")
            if worldwide_scope:
                risk_factors.append("worldwide_geographic_scope")
            if broad_activity_ban:
                risk_factors.append("broad_industry_activity_ban")

            explanation = ClauseExplanation(
                summary="Broad non-compete covenant imposing multi-year or worldwide restrictions on business activities.",
                why_it_matters="Substantially restricts the freedom to pursue business opportunities, launch products, or work with competitors.",
                supporting_evidence=" ".join(evidence) if evidence else "Restrictive covenant specifies extensive duration or global scope.",
                user_action="Negotiate to narrow the covenant: restrict solely to direct competitors for no more than 6-12 months within defined territories."
            )
        elif is_non_solicit_only or (six_months_or_less and not worldwide_scope):
            risk_level = RiskLevel.LOW
            risk_score = 2
            if is_non_solicit_only:
                risk_factors.append("limited_to_non_solicitation")
            if six_months_or_less:
                risk_factors.append("short_duration_under_6_months")

            explanation = ClauseExplanation(
                summary="Narrow restrictive covenant limited to non-solicitation or short durations.",
                why_it_matters="Focuses on preventing direct customer or employee poaching without blocking broad commercial operations.",
                supporting_evidence=" ".join(evidence) if evidence else "Narrow non-solicitation or short-term covenant.",
                user_action="Ensure human resources and sales teams are informed of the non-solicitation boundaries."
            )
        else:
            # Medium risk: standard 1-year non-compete or broad territory
            risk_level = RiskLevel.MEDIUM
            risk_score = 5
            if one_year:
                risk_factors.append("one_year_post_termination_restriction")
            if worldwide_scope:
                risk_factors.append("broad_territorial_scope")
            risk_factors.append("standard_commercial_restrictive_covenant")

            explanation = ClauseExplanation(
                summary="Moderate restrictive covenant barring competitive activities for up to 1-2 years.",
                why_it_matters="Imposes ongoing post-termination operational restrictions that may restrict future commercial projects.",
                supporting_evidence=" ".join(evidence) if evidence else "Covenant restricts competition for a standard 1-year window.",
                user_action="Seek to limit the definition of competitive business strictly to the specific project scope."
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
