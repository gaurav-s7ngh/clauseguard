"""
ClauseGuard Stage 5: Risk Assessment & Explanation Engine
=========================================================
Consumes Stage 4 clause-extraction output, executes contextual rule analysis
across all nine categories, and generates explainable, JSON-serializable
findings for Stage 6 backend consumption.
"""

from typing import Dict, Any, List, Optional, Union
import json

from ml.preprocessing.constants import LABEL_NAMES
from ml.risk_assessment.models import (
    RiskLevel,
    AssessmentStatus,
    AssessmentFinding,
    RiskSummary,
    CategoryAssessmentSummary,
    DocumentAssessment
)
from ml.risk_assessment.rules import get_analyzer_for_category, BaseClauseRuleAnalyzer


class ClauseGuardRiskEngine:
    """
    Public engine for evaluating legal contractual risk and generating structured explanations.
    """

    def __init__(self):
        # Pre-instantiate rule analyzers for efficiency
        self.analyzers = {
            cat: get_analyzer_for_category(cat) for cat in LABEL_NAMES
        }

    def assess(
        self,
        stage4_output: Dict[str, Any],
        full_contract_text: Optional[str] = None
    ) -> DocumentAssessment:
        """
        Executes end-to-end risk assessment over Stage 4 extraction output.

        Args:
            stage4_output: Dictionary output produced by Stage 4 inference pipeline.
            full_contract_text: Optional complete contract text for document-level verification.

        Returns:
            DocumentAssessment instance conforming to Stage 6 backend requirements.
        """
        # 1. Normalize and validate input structure
        if not isinstance(stage4_output, dict):
            raise ValueError(f"Expected dict input from Stage 4, got {type(stage4_output)}")

        document_id = stage4_output.get("document_id", "contract_doc")
        doc_char_len = stage4_output.get("document_char_length", len(full_contract_text or ""))
        extracted_clauses = stage4_output.get("clauses", [])
        detected_categories = set(stage4_output.get("detected_categories", []))
        extracted_categories = set(stage4_output.get("extracted_categories", []))

        # If detected_categories not explicitly provided, infer from extracted clauses
        if not detected_categories:
            detected_categories = set(cl.get("category") for cl in extracted_clauses if cl.get("category"))

        findings: List[AssessmentFinding] = []

        # 2. Assess extracted clauses
        categories_with_extracted_findings = set()
        for cl in extracted_clauses:
            cat = cl.get("category")
            if not cat or cat not in self.analyzers:
                continue

            analyzer = self.analyzers[cat]
            finding = analyzer.analyze(cl, document_context=full_contract_text)
            findings.append(finding)
            categories_with_extracted_findings.add(cat)

        # 3. Account for detected categories lacking extracted spans (zero silent discard)
        unextracted_categories = detected_categories - categories_with_extracted_findings
        for unextracted_cat in sorted(list(unextracted_categories)):
            if unextracted_cat in self.analyzers:
                fallback_finding = BaseClauseRuleAnalyzer.create_unextracted_finding(
                    category=unextracted_cat,
                    document_id=document_id
                )
                findings.append(fallback_finding)

        # 4. Compute Category-Level Summaries
        category_summary: Dict[str, CategoryAssessmentSummary] = {}
        all_relevant_categories = detected_categories.union(categories_with_extracted_findings)

        # Priority order for highest risk calculation
        risk_priority = {
            RiskLevel.HIGH: 4,
            RiskLevel.NEEDS_REVIEW: 3,
            RiskLevel.MEDIUM: 2,
            RiskLevel.LOW: 1
        }

        for cat in sorted(list(all_relevant_categories)):
            cat_findings = [f for f in findings if f.category == cat]
            clauses_count = sum(1 for f in cat_findings if f.assessment_status != AssessmentStatus.UNEXTRACTED_DETECTION)

            if cat_findings:
                highest = max(cat_findings, key=lambda f: risk_priority[f.risk_level]).risk_level
                review_req = any(f.review_required for f in cat_findings)
            else:
                highest = RiskLevel.LOW
                review_req = False

            category_summary[cat] = CategoryAssessmentSummary(
                category=cat,
                detected=(cat in detected_categories),
                clauses_extracted=clauses_count,
                highest_risk=highest,
                review_required=review_req
            )

        # 5. Compute Document-Level Risk Metrics
        high_cnt = sum(1 for f in findings if f.risk_level == RiskLevel.HIGH)
        med_cnt = sum(1 for f in findings if f.risk_level == RiskLevel.MEDIUM)
        low_cnt = sum(1 for f in findings if f.risk_level == RiskLevel.LOW)
        review_cnt = sum(1 for f in findings if f.risk_level == RiskLevel.NEEDS_REVIEW)

        # Overall risk level determination
        if high_cnt >= 1:
            overall_risk = RiskLevel.HIGH
        elif med_cnt >= 1:
            overall_risk = RiskLevel.MEDIUM
        elif review_cnt >= 1 and high_cnt == 0 and med_cnt == 0 and low_cnt == 0:
            overall_risk = RiskLevel.NEEDS_REVIEW
        else:
            overall_risk = RiskLevel.LOW

        # Defensible risk score calculation:
        # Balanced composite: 50% maximum severity + 50% average severity among scored findings
        scored_findings = [f.risk_score for f in findings if f.risk_score is not None]
        if scored_findings:
            max_score = max(scored_findings)
            mean_score = sum(scored_findings) / len(scored_findings)
            overall_score = (max_score * 0.5) + (mean_score * 0.5)
        else:
            overall_score = None

        risk_summary = RiskSummary(
            high_risk_count=high_cnt,
            medium_risk_count=med_cnt,
            low_risk_count=low_cnt,
            needs_review_count=review_cnt,
            overall_document_risk=overall_risk,
            overall_risk_score=overall_score
        )

        return DocumentAssessment(
            document_id=document_id,
            document_char_length=doc_char_len,
            total_clauses_assessed=len(findings),
            risk_summary=risk_summary,
            category_summary=category_summary,
            findings=findings,
            pipeline_version="clauseguard-stage5-v1"
        )


def assess_document_risk(
    stage4_output: Dict[str, Any],
    full_contract_text: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience public entry point for Stage 6 backend integration.
    Consumes Stage 4 output dictionary and returns a JSON-serializable assessment dictionary.
    """
    engine = ClauseGuardRiskEngine()
    assessment = engine.assess(stage4_output, full_contract_text=full_contract_text)
    return assessment.to_dict()
