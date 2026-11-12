"""
Base Clause Rule Analyzer
=========================
Defines the abstract interface and linguistic helper utilities for all
nine category-specific risk assessment rule analyzers.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import re

from ml.risk_assessment.models import (
    RiskLevel,
    AssessmentStatus,
    AssessmentFinding,
    ClauseExplanation
)


class BaseClauseRuleAnalyzer(ABC):
    """Abstract base class for all category-specific risk analyzers."""

    def __init__(self, category_name: str):
        self.category_name = category_name

    @abstractmethod
    def analyze(
        self,
        clause_dict: Dict[str, Any],
        document_context: Optional[str] = None
    ) -> AssessmentFinding:
        """
        Analyzes a single extracted clause dictionary and returns an AssessmentFinding.

        Args:
            clause_dict: Clause dictionary from Stage 4 output.
            document_context: Optional full contract text for contextual validation.

        Returns:
            Structured AssessmentFinding.
        """
        pass

    def check_minimum_evidence(self, clause_dict: Dict[str, Any]) -> Optional[AssessmentFinding]:
        """
        Verifies that clause text is sufficient for reliable risk analysis.
        Returns a Needs Review finding if text is empty, truncated, or fragmented.
        """
        text = clause_dict.get("clause_text", "")
        doc_id = clause_dict.get("document_id", "unknown_doc")
        conf = float(clause_dict.get("confidence", 0.0))
        start_char = clause_dict.get("start_char")
        end_char = clause_dict.get("end_char")
        extraction_status = clause_dict.get("extraction_status", "extracted")

        # Case 1: Empty text
        if not text or not text.strip():
            return AssessmentFinding(
                document_id=doc_id,
                category=self.category_name,
                clause_text="",
                start_char=start_char,
                end_char=end_char,
                confidence=conf,
                risk_level=RiskLevel.NEEDS_REVIEW,
                risk_score=None,
                explanation=ClauseExplanation(
                    summary="Clause text is empty or missing from extraction output.",
                    why_it_matters="Risk assessment cannot evaluate empty text.",
                    supporting_evidence="No extracted characters available for analysis.",
                    user_action="Inspect the source document manually to locate the relevant provision."
                ),
                evidence=[],
                risk_factors=["missing_clause_text"],
                extraction_status=extraction_status,
                assessment_status=AssessmentStatus.INSUFFICIENT_EVIDENCE,
                review_required=True
            )

        # Case 2: Sub-minimal text length (< 25 characters) or uninformative fragment
        cleaned = text.strip()
        if len(cleaned) < 25:
            # Check if surrounding context is available to provide more insight
            surrounding = clause_dict.get("surrounding_context", "")
            evidence_snippet = f"Fragment '{cleaned}' is too brief for definitive assessment."
            action_msg = "Review the full paragraph in the source contract to understand the complete clause."
            if surrounding and len(surrounding.strip()) > len(cleaned):
                evidence_snippet += f" Enclosing context indicates nearby text: '{surrounding[:80]}...'"

            return AssessmentFinding(
                document_id=doc_id,
                category=self.category_name,
                clause_text=cleaned,
                start_char=start_char,
                end_char=end_char,
                confidence=conf,
                risk_level=RiskLevel.NEEDS_REVIEW,
                risk_score=None,
                explanation=ClauseExplanation(
                    summary="Extracted text is an isolated fragment with insufficient context.",
                    why_it_matters="Evaluating incomplete fragments risks inaccurate legal interpretations.",
                    supporting_evidence=evidence_snippet,
                    user_action=action_msg
                ),
                evidence=[cleaned],
                risk_factors=["fragmentary_text", "uncertain_boundary"],
                extraction_status=extraction_status,
                assessment_status=AssessmentStatus.UNCERTAIN_BOUNDARY,
                review_required=True
            )

        return None

    @staticmethod
    def is_mutual(text: str) -> bool:
        """Determines if the clause applies bilaterally/mutually to both parties."""
        text_lower = text.lower()
        mutual_patterns = [
            r"\b(?:either|both|neither)\s+party\b",
            r"\beach\s+party\b",
            r"\bmutually\b",
            r"\bmutual\s+agreement\b",
            r"\bparties\s+agree\b",
            r"\bthe\s+parties\b",
            r"\bno\s+party\b"
        ]
        return any(re.search(p, text_lower) is not None for p in mutual_patterns)

    @staticmethod
    def extract_notice_days(text: str) -> Optional[int]:
        """
        Extracts notice period duration in calendar or business days.
        Returns 0 for immediate / without notice.
        """
        text_lower = text.lower()
        # Immediate termination patterns
        if re.search(r"\b(?:immediately|without\s+(?:prior\s+)?notice|at\s+any\s+time\s+without\s+notice)\b", text_lower):
            return 0

        # Numeric days pattern: "30 days", "thirty (30) days", "at least 60 calendar days"
        patterns = [
            r"(?:upon|with|at\s+least|prior|within)\s+(?:not\s+less\s+than\s+)?(?:[a-z\-]+\s*\()?(\d+)\)?\s*(?:calendar|business)?\s*days?",
            r"(\d+)\s*(?:calendar|business)?\s*days?\s*(?:prior\s+)?(?:written\s+)?notice"
        ]
        for pat in patterns:
            match = re.search(pat, text_lower)
            if match:
                try:
                    return int(match.group(1))
                except (ValueError, IndexError):
                    continue

        # Textual numbers
        text_num_map = {
            "ten": 10, "fourteen": 14, "fifteen": 15, "twenty": 20,
            "thirty": 30, "forty-five": 45, "sixty": 60, "ninety": 90, "one hundred twenty": 120
        }
        for word, val in text_num_map.items():
            if re.search(rf"\b{word}\s*(?:calendar|business)?\s*days?\b", text_lower):
                return val

        return None

    @staticmethod
    def extract_evidence_snippets(text: str, patterns: List[str]) -> List[str]:
        """Finds matched snippets corresponding to patterns for evidence citations."""
        evidence = []
        for pat in patterns:
            matches = re.finditer(pat, text, re.IGNORECASE)
            for m in matches:
                # Capture around match up to 80 chars
                s = max(0, m.start() - 10)
                e = min(len(text), m.end() + 10)
                snippet = text[s:e].strip()
                if snippet and snippet not in evidence:
                    evidence.append(snippet)
        return evidence[:4]  # Keep concise

    @classmethod
    def create_unextracted_finding(cls, category: str, document_id: str) -> AssessmentFinding:
        """Generates a standardized finding for a detected category lacking extracted spans."""
        readable_name = category.replace("_", " ").title()
        return AssessmentFinding(
            document_id=document_id,
            category=category,
            clause_text="",
            start_char=None,
            end_char=None,
            confidence=0.0,
            risk_level=RiskLevel.NEEDS_REVIEW,
            risk_score=None,
            explanation=ClauseExplanation(
                summary=f"{readable_name} clause detected in contract, but specific span boundaries were not isolated.",
                why_it_matters=f"{readable_name} terms define critical legal rights and obligations that require verification.",
                supporting_evidence="Stage 3 multi-label classifier confirmed category presence across document chunks.",
                user_action=f"Manually locate and review the {readable_name} section in the source document."
            ),
            evidence=[],
            risk_factors=["unextracted_detection"],
            extraction_status="unextracted",
            assessment_status=AssessmentStatus.UNEXTRACTED_DETECTION,
            review_required=True
        )
