"""
Anti-Assignment Risk Analyzer
=============================
Evaluates transfer restrictions, M&A / change-of-control carve-outs, consent standards, and mutuality.
"""

from typing import Dict, Any, Optional, List
import re

from ml.risk_assessment.models import RiskLevel, AssessmentStatus, AssessmentFinding, ClauseExplanation
from ml.risk_assessment.rules.base import BaseClauseRuleAnalyzer


class AntiAssignmentAnalyzer(BaseClauseRuleAnalyzer):
    """Analyzes assignment and delegation restriction provisions."""

    def __init__(self):
        super().__init__("anti_assignment")

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

        # M&A / Affiliate carve-outs
        has_ma_carveout = bool(re.search(
            r"\b(?:merger|acquisition|sale\s+of\s+(?:all\s+or\s+substantially\s+all|assets)|reorganization|change\s+of\s+control)\b",
            text_lower
        ))
        has_affiliate_carveout = bool(re.search(r"\b(?:affiliate|subsidiary)\b", text_lower))
        has_exceptions = has_ma_carveout or has_affiliate_carveout

        # Consent standard
        consent_not_unreasonable = bool(re.search(
            r"\b(?:not\s+(?:to\s+be\s+)?unreasonably\s+withheld|delayed|conditioned)\b",
            text_lower
        ))
        sole_discretion = bool(re.search(
            r"\b(?:sole\s+(?:and\s+absolute\s+)?discretion|without\s+recourse)\b",
            text_lower
        ))

        # One-sided assignment freedom
        other_can_assign = bool(re.search(
            r"\b(?:vendor|supplier|licensor|company)\s+may\s+(?:freely\s+)?assign\b",
            text_lower
        )) and not is_mutual

        # Collect evidence snippets
        if has_ma_carveout:
            evidence.extend(self.extract_evidence_snippets(text, [r"\b(?:merger|acquisition|sale\s+of\s+assets|change\s+of\s+control)\b"]))
        if consent_not_unreasonable:
            evidence.extend(self.extract_evidence_snippets(text, [r"\bnot\s+unreasonably\s+withheld\b"]))
        elif sole_discretion:
            evidence.extend(self.extract_evidence_snippets(text, [r"\bsole\s+discretion\b"]))
        if other_can_assign:
            evidence.extend(self.extract_evidence_snippets(text, [r"\b(?:vendor|supplier|licensor|company)\s+may\s+assign\b"]))

        # Decision Logic
        if other_can_assign and not has_exceptions:
            risk_level = RiskLevel.HIGH
            risk_score = 7
            risk_factors.append("unilateral_assignment_right")
            risk_factors.append("no_ma_carveout")
            explanation = ClauseExplanation(
                summary="Asymmetrical assignment restriction permitting the other party to assign while restricting user.",
                why_it_matters="Could result in contracts being assigned to a direct competitor or unwanted entity without consent.",
                supporting_evidence=" ".join(evidence) if evidence else "One party retains free assignment while the other is restricted.",
                user_action="Make assignment restrictions strictly mutual and require written consent for any third-party transfer."
            )
        elif is_mutual and has_exceptions and consent_not_unreasonable:
            risk_level = RiskLevel.LOW
            risk_score = 2
            risk_factors.append("mutual_assignment_restriction")
            risk_factors.append("ma_and_affiliate_carveout")
            risk_factors.append("reasonable_consent_standard")
            explanation = ClauseExplanation(
                summary="Balanced assignment restriction with standard exceptions for mergers, acquisitions, and affiliates.",
                why_it_matters="Protects contract stability while ensuring future corporate transactions or restructurings are not blocked.",
                supporting_evidence=" ".join(evidence) if evidence else "Mutual restriction with express merger/affiliate carve-outs.",
                user_action="Standard commercial provision; no immediate amendments required."
            )
        elif not has_exceptions:
            # Medium risk: standard anti-assignment but lacks M&A exception
            risk_level = RiskLevel.MEDIUM
            risk_score = 4
            risk_factors.append("missing_ma_carveout")
            if sole_discretion:
                risk_factors.append("sole_discretion_consent")

            explanation = ClauseExplanation(
                summary="Strict anti-assignment provision lacking express carve-outs for corporate transactions.",
                why_it_matters="If your company undergoes a future acquisition, merger, or reorganization, this contract could require third-party consent.",
                supporting_evidence=" ".join(evidence) if evidence else "Assignment barred without express exception for change of control or asset sale.",
                user_action="Add an express exception permitting assignment in connection with a merger, acquisition, or sale of substantially all assets."
            )
        else:
            risk_level = RiskLevel.LOW
            risk_score = 3
            risk_factors.append("standard_assignment_clause")
            explanation = ClauseExplanation(
                summary="Standard assignment clause with customary transfer guidelines.",
                why_it_matters="Governs contract continuity upon organizational changes.",
                supporting_evidence=" ".join(evidence) if evidence else "Assignment terms conform to standard commercial practices.",
                user_action="Confirm that designated affiliates meet the definition criteria."
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
