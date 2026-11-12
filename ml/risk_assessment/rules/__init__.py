"""
ClauseGuard Category Rule Analyzers Registry
============================================
Provides unified factory access to all nine category-specific analyzers.
"""

from typing import Dict, Type
from ml.risk_assessment.rules.base import BaseClauseRuleAnalyzer
from ml.risk_assessment.rules.liability_cap import LiabilityCapAnalyzer
from ml.risk_assessment.rules.termination import TerminationAnalyzer
from ml.risk_assessment.rules.assignment import AntiAssignmentAnalyzer
from ml.risk_assessment.rules.renewal import RenewalTermAnalyzer
from ml.risk_assessment.rules.governing_law import GoverningLawAnalyzer
from ml.risk_assessment.rules.audit import AuditRightsAnalyzer
from ml.risk_assessment.rules.ip_ownership import IPOwnershipAnalyzer
from ml.risk_assessment.rules.license_grant import LicenseGrantAnalyzer
from ml.risk_assessment.rules.non_compete import NonCompeteAnalyzer

ANALYZER_REGISTRY: Dict[str, Type[BaseClauseRuleAnalyzer]] = {
    "liability_cap": LiabilityCapAnalyzer,
    "termination_for_convenience": TerminationAnalyzer,
    "anti_assignment": AntiAssignmentAnalyzer,
    "renewal_term": RenewalTermAnalyzer,
    "governing_law": GoverningLawAnalyzer,
    "audit_rights": AuditRightsAnalyzer,
    "ip_ownership_assignment": IPOwnershipAnalyzer,
    "license_grant": LicenseGrantAnalyzer,
    "non_compete": NonCompeteAnalyzer,
}


def get_analyzer_for_category(category_name: str) -> BaseClauseRuleAnalyzer:
    """Returns an instantiated rule analyzer for the specified category."""
    if category_name not in ANALYZER_REGISTRY:
        raise ValueError(f"No risk analyzer registered for category: '{category_name}'")
    return ANALYZER_REGISTRY[category_name]()
