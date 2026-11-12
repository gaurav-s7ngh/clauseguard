"""
IP Ownership Assignment Risk Analyzer
=====================================
Evaluates intellectual property transfers, pre-existing IP retention, work-for-hire scope, and moral rights waivers.
"""

from typing import Dict, Any, Optional, List
import re

from ml.risk_assessment.models import RiskLevel, AssessmentStatus, AssessmentFinding, ClauseExplanation
from ml.risk_assessment.rules.base import BaseClauseRuleAnalyzer


class IPOwnershipAnalyzer(BaseClauseRuleAnalyzer):
    """Analyzes IP ownership, invention assignment, and work-for-hire provisions."""

    def __init__(self):
        super().__init__("ip_ownership_assignment")

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

        # Pre-existing / Background IP reservation
        has_preexisting_carveout = bool(re.search(
            r"\b(?:pre[\-\s]existing\s+(?:ip|intellectual\s+property|technology|materials)|background\s+(?:ip|technology)|each\s+party\s+retains\s+(?:all\s+right|ownership))\b",
            text_lower
        ))

        # Scope of assignment: broad vs tailored
        broad_assignment = bool(re.search(
            r"\b(?:all\s+inventions|all\s+intellectual\s+property|all\s+work\s+product|whether\s+or\s+not\s+(?:related|patentable)|sole\s+and\s+exclusive\s+property\s+of\s+company|assigns\s+all\s+(?:right,\s+title|future))\b",
            text_lower
        ))

        tailored_deliverables_only = bool(re.search(
            r"\b(?:deliverables\s+specifically\s+created|solely\s+for\s+(?:client|customer)\s+under\s+this|custom\s+deliverables)\b",
            text_lower
        ))

        # Payment condition precedent
        conditioned_on_payment = bool(re.search(
            r"\b(?:upon\s+full\s+payment|subject\s+to\s+payment\s+in\s+full|receipt\s+of\s+all\s+applicable\s+fees)\b",
            text_lower
        ))

        # Moral rights waiver
        moral_rights_waiver = bool(re.search(
            r"\b(?:waive\w*\s+(?:all\s+)?moral\s+rights|paternity|integrity)\b",
            text_lower
        ))

        # Collect evidence
        if has_preexisting_carveout:
            evidence.extend(self.extract_evidence_snippets(text, [r"\b(?:pre[\-\s]existing\s+ip|retains\s+all\s+right|background\s+technology)\b"]))
        if broad_assignment:
            evidence.extend(self.extract_evidence_snippets(text, [r"\b(?:all\s+inventions|assigns\s+all\s+right|sole\s+and\s+exclusive\s+property)\b"]))
        if moral_rights_waiver:
            evidence.extend(self.extract_evidence_snippets(text, [r"\bmoral\s+rights\b"]))

        # Decision Logic
        if broad_assignment and not has_preexisting_carveout:
            risk_level = RiskLevel.HIGH
            risk_score = 8
            risk_factors.append("overbroad_ip_assignment")
            risk_factors.append("missing_preexisting_ip_carveout")
            if moral_rights_waiver:
                risk_factors.append("unconditional_moral_rights_waiver")

            explanation = ClauseExplanation(
                summary="Broad IP assignment that lacks express protections for pre-existing or background technology.",
                why_it_matters="Risks inadvertently transferring ownership of your proprietary tools, general code libraries, or pre-existing assets.",
                supporting_evidence=" ".join(evidence) if evidence else "Assignment encompasses all inventions without carving out background IP.",
                user_action="Insert an explicit reservation of rights confirming each party retains ownership of its pre-existing IP and tools."
            )
        elif has_preexisting_carveout and (tailored_deliverables_only or not broad_assignment):
            risk_level = RiskLevel.LOW
            risk_score = 2
            risk_factors.append("preexisting_ip_expressly_retained")
            if tailored_deliverables_only:
                risk_factors.append("assignment_limited_to_custom_deliverables")
            if conditioned_on_payment:
                risk_factors.append("transfer_conditioned_on_payment")

            explanation = ClauseExplanation(
                summary="Balanced IP ownership clause with clear boundaries between background IP and project deliverables.",
                why_it_matters="Protects pre-existing proprietary assets while cleanly vesting ownership of commissioned deliverables in the paying party.",
                supporting_evidence=" ".join(evidence) if evidence else "Clear reservation of pre-existing intellectual property.",
                user_action="Ensure that project deliverables are clearly itemized in the applicable Statement of Work."
            )
        else:
            # Medium risk: standard assignment without complete clarity on reusable tools
            risk_level = RiskLevel.MEDIUM
            risk_score = 5
            if not has_preexisting_carveout:
                risk_factors.append("unclear_preexisting_ip_boundaries")
            if broad_assignment:
                risk_factors.append("broad_assignment_scope")

            explanation = ClauseExplanation(
                summary="IP assignment provision requiring clearer boundaries for reusable or pre-existing assets.",
                why_it_matters="Ambiguity over background materials can trigger ownership disputes over reusable software components or know-how.",
                supporting_evidence=" ".join(evidence) if evidence else "Assignment clause lacks explicit schedules of retained background technology.",
                user_action="Attach a schedule listing any pre-existing software, frameworks, or methodologies used in the project."
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
