"""
ClauseGuard Stage 5: Risk Assessment & Explanation Engine
=========================================================
Exports public entry points and data models for contract risk analysis.
"""

from ml.risk_assessment.models import (
    RiskLevel,
    AssessmentStatus,
    ClauseExplanation,
    AssessmentFinding,
    RiskSummary,
    CategoryAssessmentSummary,
    DocumentAssessment
)
from ml.risk_assessment.engine import ClauseGuardRiskEngine, assess_document_risk

__all__ = [
    "ClauseGuardRiskEngine",
    "assess_document_risk",
    "RiskLevel",
    "AssessmentStatus",
    "ClauseExplanation",
    "AssessmentFinding",
    "RiskSummary",
    "CategoryAssessmentSummary",
    "DocumentAssessment"
]
