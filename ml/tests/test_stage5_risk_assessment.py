"""
ClauseGuard Stage 5 Automated Test Suite: Risk Assessment & Explanation Engine
==============================================================================
Validates:
1. Rule assessment across all nine categories (High, Medium, Low, Needs Review).
2. One-sided vs mutual provisions.
3. Carve-outs and exceptions handling.
4. Ambiguous, empty, and truncated clause fragments.
5. Preservation of detected categories lacking extracted spans (zero silent discard).
6. Grounded Plain-English explanations covering all four mandatory fields.
7. Document-level risk aggregation and defensible risk scoring.
8. Complete Stage 6 JSON schema compatibility.
9. End-to-end integration consuming actual Stage 4 pipeline output.
"""

import json
import pytest

from ml.preprocessing.constants import LABEL_NAMES
from ml.risk_assessment.models import (
    RiskLevel,
    AssessmentStatus,
    AssessmentFinding,
    DocumentAssessment
)
from ml.risk_assessment.engine import ClauseGuardRiskEngine, assess_document_risk
from ml.inference.pipeline import ClauseGuardPipeline


# ============================================================================
# Test 1: Liability Cap Rules
# ============================================================================
def test_liability_cap_rules():
    engine = ClauseGuardRiskEngine()

    # High Risk: One-sided disclaimer / nominal cap
    high_cap = {
        "category": "liability_cap",
        "clause_text": "In no event shall Vendor's total aggregate liability exceed $100. Customer shall remain fully liable.",
        "start_char": 100,
        "end_char": 204,
        "confidence": 0.90,
        "document_id": "doc_test"
    }
    f_high = engine.assess({"clauses": [high_cap], "document_id": "doc_test"}).findings[0]
    assert f_high.risk_level == RiskLevel.HIGH
    assert f_high.risk_score >= 8
    assert "nominal_liability_cap" in f_high.risk_factors or "unilateral_vendor_protection" in f_high.risk_factors

    # Low Risk: Mutual cap tied to 12 months fees with customary carveouts
    low_cap = {
        "category": "liability_cap",
        "clause_text": "Neither party's total aggregate liability under this Agreement shall exceed the fees paid in the prior twelve (12) months, excluding breaches of confidentiality or gross negligence.",
        "start_char": 300,
        "end_char": 485,
        "confidence": 0.88,
        "document_id": "doc_test"
    }
    f_low = engine.assess({"clauses": [low_cap], "document_id": "doc_test"}).findings[0]
    assert f_low.risk_level == RiskLevel.LOW
    assert f_low.risk_score <= 3
    assert "mutual_balanced_cap" in f_low.risk_factors

    # Needs Review: External unattached schedule reference
    ambig_cap = {
        "category": "liability_cap",
        "clause_text": "The maximum aggregate liability of the parties shall be as set forth in Schedule B attached hereto.",
        "start_char": 500,
        "end_char": 600,
        "confidence": 0.75,
        "document_id": "doc_test"
    }
    f_ambig = engine.assess({"clauses": [ambig_cap], "document_id": "doc_test"}).findings[0]
    assert f_ambig.risk_level == RiskLevel.NEEDS_REVIEW
    assert f_ambig.risk_score is None
    assert f_ambig.review_required is True


# ============================================================================
# Test 2: Termination for Convenience Rules
# ============================================================================
def test_termination_for_convenience_rules():
    engine = ClauseGuardRiskEngine()

    # High Risk: Unilateral immediate termination or early termination penalty
    high_term = {
        "category": "termination_for_convenience",
        "clause_text": "Vendor may terminate this Agreement immediately without notice for convenience. If Customer terminates, Customer shall pay an early termination penalty fee.",
        "start_char": 100,
        "end_char": 260,
        "confidence": 0.85,
        "document_id": "doc_test"
    }
    f_high = engine.assess({"clauses": [high_term]}).findings[0]
    assert f_high.risk_level == RiskLevel.HIGH
    assert f_high.risk_score >= 8
    assert "early_termination_penalty" in f_high.risk_factors or "unilateral_termination_right" in f_high.risk_factors

    # Low Risk: Mutual 30+ days written notice
    low_term = {
        "category": "termination_for_convenience",
        "clause_text": "Either party may terminate this Agreement for convenience upon sixty (60) days prior written notice, and Customer shall receive a pro rata refund of any unearned prepaid fees.",
        "start_char": 300,
        "end_char": 480,
        "confidence": 0.92,
        "document_id": "doc_test"
    }
    f_low = engine.assess({"clauses": [low_term]}).findings[0]
    assert f_low.risk_level == RiskLevel.LOW
    assert f_low.risk_score <= 3
    assert "mutual_convenience_termination" in f_low.risk_factors


# ============================================================================
# Test 3: Anti-Assignment Rules
# ============================================================================
def test_anti_assignment_rules():
    engine = ClauseGuardRiskEngine()

    # High Risk: One-sided assignment right for other party, user completely blocked
    high_assign = {
        "category": "anti_assignment",
        "clause_text": "Vendor may freely assign this Agreement. Customer may not assign this Agreement to any third party under any circumstances.",
        "start_char": 100,
        "end_char": 225,
        "confidence": 0.84,
        "document_id": "doc_test"
    }
    f_high = engine.assess({"clauses": [high_assign]}).findings[0]
    assert f_high.risk_level == RiskLevel.HIGH
    assert "unilateral_assignment_right" in f_high.risk_factors

    # Low Risk: Mutual restriction with M&A carveout and reasonable consent standard
    low_assign = {
        "category": "anti_assignment",
        "clause_text": "Neither party may assign this Agreement without prior written consent, not to be unreasonably withheld, except in connection with a merger, acquisition, or sale of substantially all assets.",
        "start_char": 300,
        "end_char": 490,
        "confidence": 0.91,
        "document_id": "doc_test"
    }
    f_low = engine.assess({"clauses": [low_assign]}).findings[0]
    assert f_low.risk_level == RiskLevel.LOW
    assert "ma_and_affiliate_carveout" in f_low.risk_factors
    assert "reasonable_consent_standard" in f_low.risk_factors


# ============================================================================
# Test 4: Renewal Term Rules
# ============================================================================
def test_renewal_term_rules():
    engine = ClauseGuardRiskEngine()

    # High Risk: Multi-year lockin or 90-day cancellation window
    high_renew = {
        "category": "renewal_term",
        "clause_text": "This Agreement shall automatically renew for successive periods of three (3) years unless cancelled at least ninety (90) days prior to the expiration date.",
        "start_char": 100,
        "end_char": 260,
        "confidence": 0.89,
        "document_id": "doc_test"
    }
    f_high = engine.assess({"clauses": [high_renew]}).findings[0]
    assert f_high.risk_level == RiskLevel.HIGH
    assert "multi_year_automatic_renewal_lockin" in f_high.risk_factors

    # Low Risk: Manual renewal upon mutual written agreement
    low_renew = {
        "category": "renewal_term",
        "clause_text": "This Agreement shall expire at the end of the initial term and may be renewed only upon mutual written agreement of both parties.",
        "start_char": 300,
        "end_char": 435,
        "confidence": 0.95,
        "document_id": "doc_test"
    }
    f_low = engine.assess({"clauses": [low_renew]}).findings[0]
    assert f_low.risk_level == RiskLevel.LOW
    assert "mutual_written_renewal_only" in f_low.risk_factors


# ============================================================================
# Test 5: Governing Law Rules
# ============================================================================
def test_governing_law_rules():
    engine = ClauseGuardRiskEngine()

    # High Risk: Offshore haven or unilateral fee shifting
    high_law = {
        "category": "governing_law",
        "clause_text": "This Agreement shall be governed by the laws of the Cayman Islands. Customer shall pay all of Vendor's legal costs and attorneys' fees.",
        "start_char": 100,
        "end_char": 235,
        "confidence": 0.87,
        "document_id": "doc_test"
    }
    f_high = engine.assess({"clauses": [high_law]}).findings[0]
    assert f_high.risk_level == RiskLevel.HIGH

    # Low Risk: Standard commercial state (Delaware) with mutual fees
    low_law = {
        "category": "governing_law",
        "clause_text": "This Agreement shall be governed by the laws of the State of Delaware. The prevailing party shall be entitled to recover reasonable attorneys' fees.",
        "start_char": 300,
        "end_char": 455,
        "confidence": 0.94,
        "document_id": "doc_test"
    }
    f_low = engine.assess({"clauses": [low_law]}).findings[0]
    assert f_low.risk_level == RiskLevel.LOW
    assert "standard_commercial_jurisdiction" in f_low.risk_factors

    # Needs Review: Placeholder text
    blank_law = {
        "category": "governing_law",
        "clause_text": "This Agreement shall be construed in accordance with the laws of [Insert State].",
        "start_char": 500,
        "end_char": 580,
        "confidence": 0.80,
        "document_id": "doc_test"
    }
    f_blank = engine.assess({"clauses": [blank_law]}).findings[0]
    assert f_blank.risk_level == RiskLevel.NEEDS_REVIEW
    assert f_blank.review_required is True


# ============================================================================
# Test 6: Audit Rights Rules
# ============================================================================
def test_audit_rights_rules():
    engine = ClauseGuardRiskEngine()

    # High Risk: Unannounced / immediate inspection and audited party bears all costs
    high_audit = {
        "category": "audit_rights",
        "clause_text": "Vendor may conduct an unannounced audit at any time without notice. Customer shall bear all costs of such audit regardless of the result.",
        "start_char": 100,
        "end_char": 240,
        "confidence": 0.86,
        "document_id": "doc_test"
    }
    f_high = engine.assess({"clauses": [high_audit]}).findings[0]
    assert f_high.risk_level == RiskLevel.HIGH
    assert "unannounced_immediate_audits" in f_high.risk_factors

    # Low Risk: Annual audit on 30 days notice during business hours
    low_audit = {
        "category": "audit_rights",
        "clause_text": "Licensor may conduct an audit not more than once per calendar year upon thirty (30) business days prior written notice during regular business hours.",
        "start_char": 300,
        "end_char": 450,
        "confidence": 0.90,
        "document_id": "doc_test"
    }
    f_low = engine.assess({"clauses": [low_audit]}).findings[0]
    assert f_low.risk_level == RiskLevel.LOW
    assert "limited_to_annual_audit" in f_low.risk_factors


# ============================================================================
# Test 7: IP Ownership Assignment Rules
# ============================================================================
def test_ip_ownership_rules():
    engine = ClauseGuardRiskEngine()

    # High Risk: Broad assignment of all inventions without pre-existing IP carve-out
    high_ip = {
        "category": "ip_ownership_assignment",
        "clause_text": "Contractor assigns all right, title, and interest in and to all inventions, developments, and IP created during the term, whether or not related to this Agreement.",
        "start_char": 100,
        "end_char": 265,
        "confidence": 0.88,
        "document_id": "doc_test"
    }
    f_high = engine.assess({"clauses": [high_ip]}).findings[0]
    assert f_high.risk_level == RiskLevel.HIGH
    assert "overbroad_ip_assignment" in f_high.risk_factors

    # Low Risk: Retained pre-existing IP with custom deliverables assigned
    low_ip = {
        "category": "ip_ownership_assignment",
        "clause_text": "Each party retains all right and ownership in its pre-existing IP. Custom deliverables specifically created for Client under this Statement of Work shall vest in Client upon full payment.",
        "start_char": 300,
        "end_char": 485,
        "confidence": 0.91,
        "document_id": "doc_test"
    }
    f_low = engine.assess({"clauses": [low_ip]}).findings[0]
    assert f_low.risk_level == RiskLevel.LOW
    assert "preexisting_ip_expressly_retained" in f_low.risk_factors


# ============================================================================
# Test 8: License Grant Rules
# ============================================================================
def test_license_grant_rules():
    engine = ClauseGuardRiskEngine()

    # High Risk: Exclusive or revocable at will license
    high_lic = {
        "category": "license_grant",
        "clause_text": "Licensor grants Licensee a revocable license at will to access and use the software.",
        "start_char": 100,
        "end_char": 185,
        "confidence": 0.83,
        "document_id": "doc_test"
    }
    f_high = engine.assess({"clauses": [high_lic]}).findings[0]
    assert f_high.risk_level == RiskLevel.HIGH
    assert "at_will_revocable_license" in f_high.risk_factors

    # Low Risk: Non-exclusive, fully paid-up license
    low_lic = {
        "category": "license_grant",
        "clause_text": "Supplier grants Customer a non-exclusive, worldwide, fully paid-up, royalty-free license to use the documentation for internal business operations.",
        "start_char": 300,
        "end_char": 445,
        "confidence": 0.93,
        "document_id": "doc_test"
    }
    f_low = engine.assess({"clauses": [low_lic]}).findings[0]
    assert f_low.risk_level == RiskLevel.LOW
    assert "non_exclusive_license" in f_low.risk_factors


# ============================================================================
# Test 9: Non-Compete Rules
# ============================================================================
def test_non_compete_rules():
    engine = ClauseGuardRiskEngine()

    # High Risk: 3-year worldwide non-compete
    high_nc = {
        "category": "non_compete",
        "clause_text": "Contractor shall not directly or indirectly engage in any competing business worldwide for a period of three (3) years following termination.",
        "start_char": 100,
        "end_char": 240,
        "confidence": 0.89,
        "document_id": "doc_test"
    }
    f_high = engine.assess({"clauses": [high_nc]}).findings[0]
    assert f_high.risk_level == RiskLevel.HIGH
    assert "multi_year_non_compete_duration" in f_high.risk_factors
    assert "worldwide_geographic_scope" in f_high.risk_factors

    # Low Risk: Narrow non-solicit for <= 6 months
    low_nc = {
        "category": "non_compete",
        "clause_text": "For a period of six (6) months post termination, neither party shall solicit any employee of the other party.",
        "start_char": 300,
        "end_char": 410,
        "confidence": 0.90,
        "document_id": "doc_test"
    }
    f_low = engine.assess({"clauses": [low_nc]}).findings[0]
    assert f_low.risk_level == RiskLevel.LOW
    assert "limited_to_non_solicitation" in f_low.risk_factors


# ============================================================================
# Test 10: Insufficient Evidence and Fragment Handling
# ============================================================================
def test_insufficient_evidence_and_fragment_handling():
    engine = ClauseGuardRiskEngine()

    # Empty text
    empty_clause = {
        "category": "liability_cap",
        "clause_text": "",
        "start_char": 100,
        "end_char": 100,
        "confidence": 0.0,
        "document_id": "doc_empty"
    }
    f_empty = engine.assess({"clauses": [empty_clause]}).findings[0]
    assert f_empty.risk_level == RiskLevel.NEEDS_REVIEW
    assert f_empty.assessment_status == AssessmentStatus.INSUFFICIENT_EVIDENCE
    assert f_empty.review_required is True

    # Very short fragment (< 25 characters)
    frag_clause = {
        "category": "governing_law",
        "clause_text": "New York law.",
        "start_char": 50,
        "end_char": 63,
        "confidence": 0.35,
        "document_id": "doc_frag",
        "surrounding_context": "This Agreement shall be governed by New York law."
    }
    f_frag = engine.assess({"clauses": [frag_clause]}).findings[0]
    assert f_frag.risk_level == RiskLevel.NEEDS_REVIEW
    assert f_frag.assessment_status == AssessmentStatus.UNCERTAIN_BOUNDARY
    assert f_frag.review_required is True


# ============================================================================
# Test 11: Zero Silent Discard of Detected Categories
# ============================================================================
def test_unextracted_detection_preservation():
    """Verify that a category detected in Stage 4 without an extracted span is preserved."""
    engine = ClauseGuardRiskEngine()

    stage4_output = {
        "document_id": "doc_split",
        "detected_categories": ["liability_cap", "audit_rights"],
        "extracted_categories": ["liability_cap"],
        "clauses": [
            {
                "category": "liability_cap",
                "clause_text": "Neither party's aggregate liability shall exceed the fees paid in the prior twelve (12) months.",
                "start_char": 200,
                "end_char": 296,
                "confidence": 0.85
            }
        ]
    }

    assessment = engine.assess(stage4_output)
    assert len(assessment.findings) == 2

    # Verify unextracted audit_rights finding exists
    audit_finding = next(f for f in assessment.findings if f.category == "audit_rights")
    assert audit_finding.assessment_status == AssessmentStatus.UNEXTRACTED_DETECTION
    assert audit_finding.risk_level == RiskLevel.NEEDS_REVIEW
    assert audit_finding.review_required is True
    assert audit_finding.clause_text == ""
    assert audit_finding.start_char is None

    # Verify category summary correctly reflects unextracted detection
    cat_summary = assessment.category_summary["audit_rights"]
    assert cat_summary.detected is True
    assert cat_summary.clauses_extracted == 0
    assert cat_summary.highest_risk == RiskLevel.NEEDS_REVIEW
    assert cat_summary.review_required is True


# ============================================================================
# Test 12: Explanation Structure and Grounding
# ============================================================================
def test_explanation_structure_and_grounding():
    engine = ClauseGuardRiskEngine()

    clause = {
        "category": "governing_law",
        "clause_text": "This Agreement shall be governed by Delaware law and disputes resolved in Delaware courts.",
        "start_char": 100,
        "end_char": 190,
        "confidence": 0.90,
        "document_id": "doc_exp"
    }
    assessment = engine.assess({"clauses": [clause], "document_id": "doc_exp"})
    finding = assessment.findings[0]

    expl = finding.explanation
    assert len(expl.summary.strip()) > 10
    assert len(expl.why_it_matters.strip()) > 10
    assert len(expl.supporting_evidence.strip()) > 5
    assert len(expl.user_action.strip()) > 10
    assert "Delaware" in expl.supporting_evidence or "Delaware" in expl.summary


# ============================================================================
# Test 13: Document-Level Aggregation & Risk Scoring
# ============================================================================
def test_document_level_aggregation_and_risk_scoring():
    engine = ClauseGuardRiskEngine()

    stage4_output = {
        "document_id": "contract_master",
        "document_char_length": 5000,
        "detected_categories": ["liability_cap", "termination_for_convenience", "governing_law"],
        "extracted_categories": ["liability_cap", "termination_for_convenience", "governing_law"],
        "clauses": [
            {
                # High risk
                "category": "liability_cap",
                "clause_text": "In no event shall Vendor be liable for any damages whatsoever exceeding $10.",
                "start_char": 100,
                "end_char": 175,
                "confidence": 0.88,
                "document_id": "contract_master"
            },
            {
                # Medium risk
                "category": "termination_for_convenience",
                "clause_text": "Vendor may terminate this Agreement upon thirty (30) days notice.",
                "start_char": 300,
                "end_char": 365,
                "confidence": 0.85,
                "document_id": "contract_master"
            },
            {
                # Low risk
                "category": "governing_law",
                "clause_text": "This Agreement is governed by Delaware law with mutual attorney fees to the prevailing party.",
                "start_char": 500,
                "end_char": 593,
                "confidence": 0.94,
                "document_id": "contract_master"
            }
        ]
    }

    assessment = engine.assess(stage4_output)
    summary = assessment.risk_summary

    assert summary.high_risk_count == 1
    assert summary.medium_risk_count == 1
    assert summary.low_risk_count == 1
    assert summary.needs_review_count == 0
    assert summary.overall_document_risk == RiskLevel.HIGH
    assert summary.overall_risk_score is not None
    assert 5.0 <= summary.overall_risk_score <= 10.0


# ============================================================================
# Test 14: JSON Serialization & Stage 6 Contract Compliance
# ============================================================================
def test_json_serialization_and_stage6_schema_compliance():
    stage4_sample = {
        "document_id": "contract_sec_2020",
        "document_char_length": 2500,
        "total_chunks": 3,
        "total_raw_detections": 2,
        "total_extracted_clauses": 2,
        "strategy": "assisted",
        "assist_threshold": None,
        "pipeline_version": "clauseguard-stage4-v1",
        "detected_categories": ["liability_cap", "governing_law"],
        "extracted_categories": ["liability_cap", "governing_law"],
        "clauses": [
            {
                "category": "liability_cap",
                "clause_text": "Neither party's aggregate liability shall exceed the fees paid in the prior twelve (12) months.",
                "start_char": 150,
                "end_char": 246,
                "confidence": 0.92,
                "chunk_id": "chunk_0",
                "document_id": "contract_sec_2020",
                "page_number": None,
                "surrounding_context": "...Section 10. Neither party's aggregate liability shall exceed the fees paid in the prior twelve (12) months...",
                "model_version": "roberta-span-v1",
                "extraction_status": "extracted"
            }
        ]
    }

    result_dict = assess_document_risk(stage4_sample)

    # Must be 100% JSON serializable
    json_str = json.dumps(result_dict)
    reloaded = json.loads(json_str)

    # Document-level schema verification
    assert reloaded["document_id"] == "contract_sec_2020"
    assert "total_clauses_assessed" in reloaded
    assert "risk_summary" in reloaded
    assert "category_summary" in reloaded
    assert "findings" in reloaded

    # Clause-level finding schema verification
    finding = reloaded["findings"][0]
    required_clause_fields = {
        "document_id", "category", "clause_text", "start_char", "end_char",
        "confidence", "risk_level", "risk_score", "explanation", "evidence",
        "risk_factors", "extraction_status", "assessment_status", "review_required"
    }
    assert required_clause_fields.issubset(set(finding.keys()))

    explanation = finding["explanation"]
    assert {"summary", "why_it_matters", "supporting_evidence", "user_action"}.issubset(set(explanation.keys()))


# ============================================================================
# Test 15: End-to-End Pipeline Execution
# ============================================================================
def test_e2e_integration_with_stage4_pipeline_output():
    """Verify that actual ClauseGuardPipeline output directly feeds into Stage 5 engine."""
    pipeline = ClauseGuardPipeline(
        classifier_path="ml/checkpoints/roberta/best_checkpoint",
        extractor_path="ml/checkpoints/span_extractor/best_checkpoint",
        strategy="independent"
    )

    sample_doc = (
        "CONFIDENTIAL COMMERCIAL AGREEMENT\n\n"
        "1. Termination. Either party may terminate this Agreement for convenience upon sixty (60) days prior written notice.\n\n"
        "2. Governing Law. This Agreement shall be governed by Delaware law with mutual legal fees to the prevailing party.\n\n"
        "3. Limitation of Liability. In no event shall Supplier's total aggregate liability exceed $50."
    )

    stage4_result = pipeline.analyze_document(sample_doc, document_id="e2e_contract_01")
    assert "clauses" in stage4_result

    stage5_result = assess_document_risk(stage4_result, full_contract_text=sample_doc)

    assert stage5_result["document_id"] == "e2e_contract_01"
    assert "findings" in stage5_result
    assert stage5_result["total_clauses_assessed"] >= 1
    assert "risk_summary" in stage5_result
