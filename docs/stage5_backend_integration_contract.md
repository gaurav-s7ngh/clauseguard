# ClauseGuard Stage 5 to Stage 6 Backend Integration Contract

## 1. Overview & Team Boundary

This document defines the interface and data exchange contract between **Stage 5 (Risk Assessment & Explanation Engine)** and **Stage 6 (FastAPI Backend & PostgreSQL Persistence)**.

> [!IMPORTANT]
> **Strict Team Boundary:**
> - Stage 5 code lives exclusively inside `ml/risk_assessment/`.
> - Stage 6 developers own `backend/`, FastAPI routers, Pydantic request/response models, SQLAlchemy database models, and Alembic migrations.
> - Stage 6 developers import Stage 5 as a standard Python module via the canonical entry point defined below.

---

## 2. Public Python API Entry Point

Stage 6 backend services call Stage 5 through a single synchronous function:

```python
from ml.risk_assessment import assess_document_risk

# Basic usage consuming Stage 4 pipeline output
assessment_dict = assess_document_risk(stage4_pipeline_output)

# Optional usage providing full contract text for contextual surrounding evidence
assessment_dict = assess_document_risk(
    stage4_pipeline_output=stage4_pipeline_output,
    full_contract_text=raw_contract_text
)
```

### Class-Based Interface
For long-lived backend workers (e.g., Celery tasks, background workers), the engine can be instantiated once to reuse rule registries:

```python
from ml.risk_assessment.engine import ClauseGuardRiskEngine

engine = ClauseGuardRiskEngine()
document_assessment = engine.assess(
    stage4_output=stage4_pipeline_output,
    full_contract_text=raw_contract_text
)
assessment_dict = document_assessment.to_dict()
```

---

## 3. Input Specification (Stage 4 Output)

Stage 5 consumes the exact dictionary returned by `ClauseGuardPipeline.analyze_document()`.

### JSON Schema (Input)
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Stage4PipelineOutput",
  "type": "object",
  "required": ["document_id", "clauses"],
  "properties": {
    "document_id": {
      "type": "string",
      "description": "Unique identifier for the contract or document."
    },
    "document_char_length": {
      "type": "integer",
      "description": "Total character length of the raw document."
    },
    "total_chunks": {
      "type": "integer",
      "description": "Total text chunks processed by the pipeline."
    },
    "total_raw_detections": {
      "type": "integer",
      "description": "Count of clause spans before deduplication and merging."
    },
    "total_extracted_clauses": {
      "type": "integer",
      "description": "Count of unique extracted clauses after merging."
    },
    "strategy": {
      "type": "string",
      "enum": ["assisted", "independent"],
      "description": "Inference gating strategy used by Stage 4."
    },
    "pipeline_version": {
      "type": "string",
      "description": "Version string of the inference pipeline."
    },
    "detected_categories": {
      "type": "array",
      "items": { "type": "string" },
      "description": "Categories detected by Stage 3 classifier."
    },
    "extracted_categories": {
      "type": "array",
      "items": { "type": "string" },
      "description": "Categories with at least one extracted span."
    },
    "clauses": {
      "type": "array",
      "description": "List of extracted localized clause spans.",
      "items": {
        "type": "object",
        "required": ["category", "clause_text", "start_char", "end_char", "confidence"],
        "properties": {
          "category": { "type": "string" },
          "clause_text": { "type": "string" },
          "start_char": { "type": "integer" },
          "end_char": { "type": "integer" },
          "confidence": { "type": "number", "minimum": 0.0, "maximum": 1.0 },
          "chunk_id": { "type": "string" },
          "document_id": { "type": "string" },
          "extraction_status": { "type": "string" }
        }
      }
    }
  }
}
```

---

## 4. Output Specification (Stage 5 Assessment)

Stage 5 returns a 100% JSON-serializable dictionary (`DocumentAssessment.to_dict()`) ready for FastAPI responses or database insertion.

### Top-Level Document Assessment Schema
```json
{
  "document_id": "contract_sec_2026_01",
  "document_char_length": 4500,
  "analyzed_at": "2026-11-12T20:30:00.000000Z",
  "total_clauses_assessed": 2,
  "risk_summary": {
    "overall_document_risk": "High",
    "overall_risk_score": 7.6,
    "high_risk_count": 1,
    "medium_risk_count": 1,
    "low_risk_count": 0,
    "needs_review_count": 0,
    "review_recommended": true
  },
  "category_summary": {
    "liability_cap": {
      "category": "liability_cap",
      "detected": true,
      "clauses_extracted": 1,
      "highest_risk": "High",
      "review_required": false
    },
    "termination_for_convenience": {
      "category": "termination_for_convenience",
      "detected": true,
      "clauses_extracted": 1,
      "highest_risk": "Medium",
      "review_required": false
    },
    "anti_assignment": {
      "category": "anti_assignment",
      "detected": false,
      "clauses_extracted": 0,
      "highest_risk": null,
      "review_required": false
    }
  },
  "findings": [
    {
      "finding_id": "finding_liability_cap_150_230",
      "document_id": "contract_sec_2026_01",
      "category": "liability_cap",
      "clause_text": "In no event shall Vendor's total aggregate liability exceed $50.",
      "start_char": 150,
      "end_char": 214,
      "confidence": 0.88,
      "risk_level": "High",
      "risk_score": 8.5,
      "explanation": {
        "summary": "This limitation of liability clause imposes an extremely low nominal dollar cap ($50) on damages.",
        "why_it_matters": "A nominal cap effectively eliminates any practical financial remedy in the event of a material breach, data loss, or service failure.",
        "supporting_evidence": "exceed $50",
        "user_action": "Propose replacing the nominal dollar cap with a mutual cap tied to fees paid in the prior twelve (12) months."
      },
      "evidence": "exceed $50",
      "risk_factors": ["nominal_dollar_cap_under_100"],
      "extraction_status": "extracted",
      "assessment_status": "assessed",
      "review_required": false
    }
  ]
}
```

---

## 5. Field Definitions & Value Types

### 5.1 Enums & Standards
- `risk_level`: String enum: `"Low"`, `"Medium"`, `"High"`, `"Needs Review"`.
- `extraction_status`: String enum: `"extracted"`, `"not_extracted"`, `"partial"`.
- `assessment_status`: String enum: `"assessed"`, `"review_required"`, `"insufficient_evidence"`, `"unextracted_detection"`.
- `confidence`: Float between $0.00$ and $1.00$ (Stage 4 model confidence, decoupled from risk).
- `risk_score`: Float between $1.0$ and $10.0$.

### 5.2 The 4-Part Explanation Object
Every finding includes an `explanation` object containing:
1. `summary` (`str`): Clear, plain-English summary of what the clause does.
2. `why_it_matters` (`str`): Commercial and operational explanation of the legal risks.
3. `supporting_evidence` (`str`): Verbatim substring from the clause justifying the score.
4. `user_action` (`str`): Concrete redline recommendation for contract negotiation.

---

## 6. Edge Cases & Special Conditions

### 6.1 Preserving Unextracted Classifier Detections
When Stage 3 detects a category as present (`detected_categories`) but Stage 4 fails to localize a span (`extracted_categories` lacks the category):
- Stage 5 generates an unextracted finding:
  ```json
  {
    "category": "anti_assignment",
    "clause_text": "",
    "start_char": null,
    "end_char": null,
    "confidence": 0.0,
    "risk_level": "Needs Review",
    "risk_score": 5.0,
    "extraction_status": "not_extracted",
    "assessment_status": "review_required",
    "review_required": true
  }
  ```
- `category_summary["anti_assignment"]["review_required"]` is set to `true`.
- The UI should display a prompt alerting the user to manually verify the clause in the document.

### 6.2 Fragmentary or Noisy Spans
If an extracted span has fewer than 15 characters or contains only OCR artifacts:
- `assessment_status` is set to `"insufficient_evidence"`.
- `risk_level` defaults to `"Needs Review"`.
- `review_required` is set to `true`.

---

## 7. Recommended Database Schema for Stage 6 (PostgreSQL)

To facilitate smooth database integration without schema revisions, Stage 6 developers can use the following relational layout:

### Recommended Tables

```sql
-- Main Contract Records
CREATE TABLE contracts (
    id VARCHAR(128) PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    char_length INT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Contract Analysis Runs
CREATE TABLE analysis_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    contract_id VARCHAR(128) REFERENCES contracts(id) ON DELETE CASCADE,
    overall_document_risk VARCHAR(32) NOT NULL,
    overall_risk_score NUMERIC(3, 1) NOT NULL,
    risk_summary JSONB NOT NULL,
    category_summary JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Individual Clause Findings
CREATE TABLE clause_findings (
    id VARCHAR(128) PRIMARY KEY,
    analysis_run_id UUID REFERENCES analysis_runs(id) ON DELETE CASCADE,
    category VARCHAR(64) NOT NULL,
    risk_level VARCHAR(32) NOT NULL,
    risk_score NUMERIC(3, 1) NOT NULL,
    confidence NUMERIC(4, 3) NOT NULL,
    start_char INT,
    end_char INT,
    clause_text TEXT,
    explanation JSONB NOT NULL,
    risk_factors TEXT[] NOT NULL,
    extraction_status VARCHAR(32) NOT NULL,
    review_required BOOLEAN NOT NULL DEFAULT FALSE
);

-- Recommended Indexes
CREATE INDEX idx_findings_run_id ON clause_findings(analysis_run_id);
CREATE INDEX idx_findings_category ON clause_findings(category);
CREATE INDEX idx_findings_risk_level ON clause_findings(risk_level);
```
