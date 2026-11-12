# CLAUSEGUARD — MASTER AI PROJECT CONTEXT DOCUMENT

> **Document Type:** Master Architectural & Technical Context Document  
> **Target Audience:** Software Architects, NLP/ML Engineers, and Autonomous AI Coding Agents  
> **Repository:** `clauseguard`  
> **Primary Maintainer Team:** ClauseGuard Academic NLP Project (4-Member Team)  
> **Document Status:** Active & Authoritative  
> **Last Updated:** November 12, 2026  
> **Operating Environment:** Windows / Python 3.13.5 (CPython) / Localhost  

---

## TABLE OF CONTENTS
1. [Project Identity & Core Vision](#1-project-identity--core-vision)
2. [Complete Development History (Stages 1–4)](#2-complete-development-history-stages-14)
3. [Current Technical Architecture](#3-current-technical-architecture)
4. [Technology Stack](#4-technology-stack)
5. [Important Architectural Decisions](#5-important-architectural-decisions)
6. [Remaining Development Roadmap (Stages 5–9)](#6-remaining-development-roadmap-stages-59)
7. [Repository and Environment Context](#7-repository-and-environment-context)
8. [Known Issues, Performance Audits & Technical Debt](#8-known-issues-performance-audits--technical-debt)
9. [Instructions for AI Assistants Working on ClauseGuard](#9-instructions-for-ai-assistants-working-on-clauseguard)
10. [Next Development Stage](#10-next-development-stage)

---

## 1. PROJECT IDENTITY & CORE VISION

### 1.1 Purpose & Mission `[VERIFIED]`
ClauseGuard is an AI-powered legal document analysis system engineered to analyze commercial contracts, software Terms of Service (ToS), and corporate agreements. It automatically identifies, extracts, evaluates, and explains concerning, one-sided, or legally burdensome contractual clauses.

Contracts and online Terms of Service frequently contain hidden, high-impact risks—such as aggressive liability caps (\$0 liability limits), unilateral termination rights, perpetual non-compete covenants, forced IP ownership transfers, or automatic renewal traps. Non-lawyers, consumers, freelancers, and small business operators regularly sign or click through agreements without the means to detect or interpret these hazards.

ClauseGuard bridges this gap by transforming dense, opaque legal boilerplate into structured, evidence-grounded risk assessments accompanied by plain-English explanations.

### 1.2 Target Users `[VERIFIED]`
1. **Consumers & SaaS Subscribers:** Reviewing online service agreements, subscription ToS, and privacy disclosures before sign-up.
2. **Freelancers & Independent Contractors:** Auditing client agreements for predatory IP assignment clauses, extreme non-compete terms, or harsh indemnification/audit burdens.
3. **Small Business Operators:** Reviewing commercial vendor, supplier, and master services agreements (MSAs) without requiring full-time legal counsel.

### 1.3 The Nine Supported CUAD Clause Categories `[VERIFIED]`
ClauseGuard intentionally focuses on a robust, high-signal taxonomy of **nine canonical clause categories** selected from the Contract Understanding Atticus Dataset (CUAD v1):

| # | Product Category Key | CUAD Category Source | Risk Profile | Rationale & User Impact |
|---|---|---|---|---|
| 1 | `liability_cap` | `Cap on Liability` | High | Shields providers from consequential damages or limits liability to nominal amounts (\$0 or fees paid), leaving the user completely unprotected. |
| 2 | `termination_for_convenience` | `Termination for Convenience` | High | Grants one or both parties discretionary rights to terminate the agreement at will without demonstrating breach or cause. |
| 3 | `anti_assignment` | `Anti-Assignment` | Medium | Restricts transfer or assignment of contractual rights or business assets without counterparty consent. |
| 4 | `renewal_term` | `Renewal Term` | High | Automatically extends agreement terms and recurring billing unless canceled within a narrow, specific notice window. |
| 5 | `governing_law` | `Governing Law` | Low–Medium | Establishes choice of substantive state/national law and judicial jurisdiction, dramatically affecting dispute costs. |
| 6 | `audit_rights` | `Audit Rights` | Medium–High | Compels user/vendor to grant counterparty on-site access to books, records, servers, or facilities. |
| 7 | `ip_ownership_assignment` | `IP Ownership Assignment` | High | Forcibly transfers ownership of created work product, code, trademarks, or trade secrets to the counterparty. |
| 8 | `license_grant` | `License Grant` | Medium | Defines operational scope, revocability, exclusivity, and geographic bounds of software or content permissions. |
| 9 | `non_compete` | `Non-Compete` | High | Restricts post-termination business activity, client engagement, or commercial operations in designated markets. |

### 1.4 End-to-End Processing Workflow `[VERIFIED]`
ClauseGuard enforces a strictly modular, sequential pipeline:

```
[Legal Document]
       │
       ▼
[1. Document Ingestion & Chunking]
  - Normalizes text without mutating raw character positions.
  - Generates token-aware, paragraph-respecting chunk windows (250–350 tokens).
       │
       ▼
[2. Chunk-Level Clause Detection] (Stage 3 Baseline Classifier)
  - Fine-tuned RoBERTa-base multi-label classifier.
  - Predicts independent category probabilities across all 9 target labels.
  - Evaluates against calibrated per-category decision thresholds.
       │
       ▼
[3. Token-Level Span Extraction] (Stage 4 Multi-Channel BIO Extractor)
  - 9 independent BIO channels ($O=0, B=1, I=2$).
  - Subword-to-character offset alignment maps token tags back to exact contract substrings.
  - Cross-chunk span merging resolves boundary crossings and deduplicates adjacent windows.
       │
       ▼
[4. Contextual Risk Assessment] (Stage 5 — Planned)
  - Evaluates extracted clauses against category-specific legal heuristics and risk guidelines.
  - Decoupled from detection confidence: High confidence $\neq$ High risk.
  - Generates risk ratings (`CRITICAL`, `HIGH`, `MEDIUM`, `LOW`, `NEEDS_REVIEW`).
       │
       ▼
[5. Plain-English Explanation & Remediation] (Stage 5 — Planned)
  - Synthesizes non-lawyer-friendly explanations of why the clause is concerning.
  - Suggests concrete negotiation counter-proposals and redline language.
       │
       ▼
[6. Local API & Interactive UI] (Stages 6 & 7 — Planned)
  - FastAPI backend storing contract records in local PostgreSQL.
  - React/TypeScript/Vite local frontend displaying contract heatmaps, side-by-side risk cards, and text highlights.
```

---

## 2. COMPLETE DEVELOPMENT HISTORY (STAGES 1–4)

### Stage 1: CUAD Dataset Reconnaissance & Exploration `[VERIFIED COMPLETE]`
* **Objective:** Establish the empirical foundation of CUAD v1, evaluate file schemas, calculate category distributions, and define the MVP taxonomy.
* **Implementation:** Authored `ml/preprocessing/download_cuad.py` and `ml/preprocessing/explore_cuad.py`. Downloaded CUAD v1 from official sources into `ml/data/raw/cuad/`.
* **Findings:**
  * CUAD contains 510 contracts (408 official train, 102 official test) formatted in SQuAD-v2 JSON style.
  * Extracted 13,310 total annotations spanning 41 categories.
  * Extreme label sparsity was uncovered: 16 of the 41 categories appear in fewer than 15% of contracts.
  * Three critical product concepts are entirely missing from CUAD: general *Indemnification*, *Mandatory Arbitration / Class Action Waivers*, and *Confidentiality / NDA*.
* **Deliverables:**
  * `docs/dataset_exploration_report.md`
  * `docs/dataset_statistics.json`
  * `docs/clauseguard_category_mapping.md`
  * `docs/training_data_preparation_plan.md`

### Stage 2: Dataset Preprocessing & Training Data Pipeline `[VERIFIED COMPLETE]`
* **Objective:** Construct a reproducible, leak-free preprocessing pipeline converting raw CUAD contracts into tokenizer-independent Parquet datasets for the 9 selected categories.
* **Implementation:**
  * `ml/preprocessing/constants.py`: Canonical label constants, indices, and random seeds.
  * `ml/preprocessing/splitter.py`: Deterministic contract-level split (340 train, 68 validation, 102 official held-out test). Zero document overlap guaranteed.
  * `ml/preprocessing/chunking.py`: Paragraph-aware chunker with sentence-level fallback for oversized paragraphs (>360 words) and span lookahead.
  * `ml/preprocessing/labeling.py`: 3-state label semantics (Positive `1`, Negative `0`, Unknown `0` with supervision mask `0`).
  * `ml/preprocessing/dataset.py`: PyTorch `ClauseGuardDataset` with dynamic tokenization.
* **Deliverables:**
  * `ml/data/processed/train_chunks.parquet` (13,965 chunks; 1,936 positive; 2,933 gold spans)
  * `ml/data/processed/val_chunks.parquet` (2,812 chunks; 436 positive; 626 gold spans)
  * `ml/data/processed/test_chunks.parquet` (3,721 chunks; 575 positive; 798 gold spans)
  * `docs/preprocessing_pipeline.md` & `docs/preprocessing_validation_report.md`
  * `ml/tests/test_preprocessing_pipeline.py` (10/10 automated tests passing).

### Stage 3: Multi-Label Transformer Training & Model Comparison `[VERIFIED COMPLETE]`
* **Objective:** Fine-tune and evaluate two candidate transformer architectures on the 9-category chunk classification task, select the superior model on validation Macro F1, calibrate decision thresholds, and evaluate once on the official test set.
* **Implementation:**
  * Baseline: `roberta-base` vs Challenger: `law-ai/InLegalBERT`.
  * Multi-label masked binary cross-entropy loss with positive class weighting.
  * Evaluated on full validation split (2,812 chunks) and held-out test split (3,721 chunks).
* **Results:**
  * **Validation Macro F1:** RoBERTa-base achieved **0.2314** (calibrated) vs InLegalBERT **0.2032**. RoBERTa selected as winning baseline.
  * **Held-out Test Macro F1:** RoBERTa scored **0.2055** (calibrated) vs 0.1251 at default 0.50 threshold (+64.3% relative gain from calibration).
* **Deliverables:**
  * Checkpoint preserved: `ml/checkpoints/roberta/best_checkpoint/` (DO NOT MODIFY).
  * `evaluation/stage3_model_comparison.md` & `evaluation/stage3_model_comparison.json`.
  * `ml/tests/test_stage3_training.py` (10/10 automated tests passing).

### Stage 4: Token-Level Span Extraction & Inference Pipeline `[IN PROGRESS / UNDER AUDIT]`
* **Objective:** Implement token-level clause boundary localization supporting overlapping annotations across all 9 categories, transfer RoBERTa encoder weights, build a unified inference pipeline, and evaluate on held-out test data.
* **Implementation:**
  * `ml/extraction/token_tagging.py`: 9-channel BIO alignment, span decoding, cross-chunk merging.
  * `ml/extraction/model.py`: `ClauseGuardSpanExtractor` (RoBERTa backbone + 27-output linear head).
  * `ml/extraction/dataset.py`: `PretokenizedSpanDataset`.
  * `ml/extraction/train.py`: Extractor training engine.
  * `ml/inference/pipeline.py`: Unified `ClauseGuardPipeline` with Section 6 schema.
  * `ml/tests/test_stage4_extraction.py` (11/11 automated tests passing).
* **Audit & Status Issue:**
  * Official held-out test evaluation produced:
    * Exact-match Macro F1: **0.0000**
    * Partial-match Macro F1 (IoU $\ge$ 0.50): **0.0106**
    * Predictions: 4,186 | Gold annotations: 798
  * **Status:** Implementation and architectural plumbing are complete, but extraction quality is unacceptable. A systematic validation-only performance audit is underway. Model is **NOT** ready for Stage 5.

---

## 3. CURRENT TECHNICAL ARCHITECTURE

### 3.1 Data Ingestion & Preprocessing Architecture `[VERIFIED]`
```
Raw Contract Text (String)
       │
       ▼
[splitter.py] ──> Disjoint Contract Partitioning (Train: 340, Val: 68, Test: 102)
       │
       ▼
[chunking.py] ──> Natural Paragraph Splitting & Word Accumulation
                  Target: 250–350 tokens (180–260 words) per chunk.
                  Boundary Lookahead: Absorbs subsequent paragraph to keep 98.49% of clauses whole.
                  Preserves global `char_start` and `char_end`.
       │
       ▼
[labeling.py] ──> Multi-Hot Vectors (9 binary labels) + Supervision Masks (9 binary masks).
                  Computes `local_start` and `local_end` within chunk text.
       │
       ▼
[Parquet Storage] ──> `train_chunks.parquet`, `val_chunks.parquet`, `test_chunks.parquet`
```

### 3.2 Stage 3 Classifier Architecture `[VERIFIED]`
* **Backbone:** Pretrained `roberta-base` (12 layers, 768 hidden, 12 attention heads).
* **Classification Head:**
  $$\mathbf{h}_{\text{CLS}} = \text{Encoder}(\mathbf{x})[0]$$
  $$\mathbf{z} = \mathbf{W}_2 \cdot \text{GELU}(\mathbf{W}_1 \cdot \text{Dropout}(\mathbf{h}_{\text{CLS}}) + \mathbf{b}_1) + \mathbf{b}_2 \in \mathbb{R}^9$$
* **Loss Function:** Masked Binary Cross-Entropy with category-level positive weighting:
  $$\mathcal{L} = \frac{\sum_{k=0}^8 m_k \cdot \text{BCEWithLogitsLoss}(z_k, y_k, \text{pos\_weight}_k)}{\sum_{k=0}^8 m_k}$$

### 3.3 Stage 4 Span Extractor Architecture `[VERIFIED]`
* **Backbone:** Reuses fine-tuned Stage 3 RoBERTa encoder (`ml/checkpoints/roberta/best_checkpoint/`).
* **Multi-Channel Token Classification Head:**
  A single linear projection produces logits for all 9 categories simultaneously without mutual exclusion:
  $$\mathbf{H} = \text{Encoder}(\mathbf{x}) \in \mathbb{R}^{B \times L \times 768}$$
  $$\mathbf{Z} = \text{Dropout}(\mathbf{H}) \cdot \mathbf{W}^T + \mathbf{b} \in \mathbb{R}^{B \times L \times (9 \times 3)}$$
  Reshaped to:
  $$\mathbf{Z} \in \mathbb{R}^{B \times L \times 9 \times 3}$$
  Where for each token $t_i$ and category $k$:
  $$\hat{y}_{i, k} \in \{O = 0, B = 1, I = 2\}$$
* **Loss Function:** Masked Cross-Entropy Loss with class weights $\mathbf{w} = [w_O, w_B, w_I]$:
  $$\mathcal{L}_k = \text{CrossEntropyLoss}(\mathbf{Z}_{:, :, k, :}, \mathbf{Y}_{:, :, k}, \text{weight}=\mathbf{w}, \text{ignore\_index}=-100)$$

### 3.4 Integration & Inference Pipeline (`ClauseGuardPipeline`) `[VERIFIED]`
Implemented in `ml/inference/pipeline.py`:
1. **Document Chunking:** Chunks contract text into paragraph windows.
2. **Strategy Selection:**
   * `strategy="independent"`: Runs 9-channel span extraction across all chunks.
   * `strategy="assisted"`: Runs Stage 3 classifier first; only activates channels meeting probability threshold ($\tau \ge 0.20$).
3. **Span Decoding & Cross-Chunk Merging:** Converts token BIO sequences to local character offsets, projects to global document coordinates, and merges duplicate or split spans across chunk boundaries.

### 3.5 Output Schema Contract (Stage 5 Ingestion Standard) `[VERIFIED]`
Every extracted clause strictly adheres to the following JSON schema:
```json
{
  "document_id": "contract_sec_2020",
  "document_char_length": 14250,
  "total_chunks": 12,
  "total_raw_detections": 8,
  "total_extracted_clauses": 5,
  "strategy": "independent",
  "pipeline_version": "clauseguard-stage4-v1",
  "clauses": [
    {
      "category": "liability_cap",
      "clause_text": "IN NO EVENT SHALL SUPPLIER'S TOTAL AGGREGATE LIABILITY EXCEED THE TOTAL FEES PAID HEREUNDER IN THE PRIOR TWELVE (12) MONTHS.",
      "start_char": 4512,
      "end_char": 4635,
      "confidence": 0.8245,
      "chunk_id": "chunk_004",
      "document_id": "contract_sec_2020",
      "page_number": null,
      "surrounding_context": "...Section 11. Limitation of Liability. IN NO EVENT SHALL SUPPLIER'S TOTAL AGGREGATE LIABILITY EXCEED THE TOTAL FEES PAID HEREUNDER IN THE PRIOR TWELVE (12) MONTHS. This limitation applies regardless of the legal theory...",
      "model_version": "roberta-span-v1",
      "extraction_status": "extracted"
    }
  ]
}
```

---

## 4. TECHNOLOGY STACK

| Technology | Version / Spec | Role in ClauseGuard | Relevant Stages |
|---|---|---|---|
| **Python** | `3.13.5` | Core programming language for ML pipeline, backend, and scripts. | Stages 1–9 |
| **PyTorch** | `2.6.0+cpu` | Deep learning computation, tensor operations, backpropagation. | Stages 2–4 |
| **Transformers** | `4.49.0` | Hugging Face transformer backbones, tokenizers, learning rate schedulers. | Stages 3–4 |
| **RoBERTa-base** | Hugging Face: `roberta-base` | Primary baseline backbone encoder (125M parameters). | Stages 3–4 |
| **InLegalBERT** | Hugging Face: `law-ai/InLegalBERT` | Legal domain challenger evaluated in Stage 3. | Stage 3 |
| **Pandas & PyArrow** | `2.2.3` / `19.0.0` | High-throughput columnar chunk storage (`.parquet`) and dataset manipulation. | Stages 1–4 |
| **Scikit-learn** | `1.6.1` | Train/val splitting, precision/recall/F1 metrics, calibration curves. | Stages 2–4 |
| **FastAPI** | `0.115.8` `[PLANNED]` | High-performance asynchronous REST API framework for local serving. | Stage 6 |
| **Uvicorn** | `0.34.0` `[PLANNED]` | ASGI production server for FastAPI. | Stage 6 |
| **PostgreSQL** | `15+ / 16+` `[PLANNED]` | Relational database for contract metadata, extraction runs, and risk logs. | Stage 6 |
| **SQLAlchemy / Alembic**| `2.0.38` / `1.14.1` `[PLANNED]`| Database ORM and schema migration manager. | Stage 6 |
| **React** | `^19.3.1` `[PLANNED]` | Component-based interactive user interface framework. | Stage 7 |
| **TypeScript** | `~6.0.3` `[PLANNED]` | Type-safe frontend client implementation. | Stage 7 |
| **Vite** | `^8.3.6` `[PLANNED]` | Next-generation frontend build tool and local dev server. | Stage 7 |
| **Pytest** | `9.2.0` | Automated testing framework for regression and integration suites. | Stages 2–8 |

---

## 5. IMPORTANT ARCHITECTURAL DECISIONS

The following technical decisions are foundational. Future AI assistants and developers **must not reverse or modify them** without explicit documented authorization:

1. **CUAD as the Primary Training Corpus:**  
   CUAD v1 represents the highest-quality, expert-annotated legal contract dataset available. It provides the grounding for the initial 9 categories.
2. **Contract-Level Dataset Splitting (Zero Document Leakage):**  
   Splits are strictly partitioned at the contract title level (340 train, 68 validation, 102 test). Chunks from the same contract never appear in multiple splits.
3. **RoBERTa-base Selected as Stage 3 Baseline:**  
   RoBERTa-base outperformed InLegalBERT on validation Macro F1 (0.2314 vs 0.2032) and generalizes better to US commercial agreements.
4. **Stage 3 Baseline Checkpoint is FROZEN:**  
   `ml/checkpoints/roberta/best_checkpoint/` is an immutable historical baseline. It must never be retrained or overwritten.
5. **Multi-Channel BIO Scheme for Overlapping Annotations:**  
   Flat 19-class BIO is prohibited. All token extraction must use 9 independent 3-way BIO channels ($O, B, I$) to support multiple and overlapping clauses.
6. **Masked Supervision for Unverified Annotations:**  
   In CUAD, unverified categories for a contract must be assigned label `-100` (`IGNORE_INDEX`). They must never default to negative ($O$) supervision.
7. **Exact Original Character-Offset Preservation:**  
   Raw text is never mutated, stripped, or lowercased prior to indexing. Every token offset and extracted span must map to exact slices of the original contract string.
8. **Decoupling Detection Confidence from Risk Severity:**  
   A model predicting a clause with 0.95 confidence does not imply the clause is 95% risky. Confidence represents detection certainty; risk assessment is an independent semantic evaluation.
9. **Detection Does Not Imply Dangerousness:**  
   A standard 30-day termination clause is routine; an immediate unilateral termination clause is hazardous. Presence alone does not dictate risk.
10. **Evidence-Grounded, Category-Specific Risk Logic:**  
    Risk assessment must be grounded in explicit textual patterns, liabilities, and legal exposure guidelines, not generic LLM speculation.
11. **Explicit Needs-Review / Fallback States:**  
    When a clause is ambiguous, truncated, or outside calibration bounds, the system must output `"NEEDS_REVIEW"` or `"UNABLE_TO_ASSESS"` rather than hallucinating safety.
12. **Local Execution Priority:**  
    ClauseGuard is designed to operate locally on user hardware (FastAPI + local PostgreSQL + Vite UI). Cloud deployment or external SaaS dependencies are outside current scope.

---

## 6. REMAINING DEVELOPMENT ROADMAP (STAGES 5–9)

### Stage 5: Contextual Risk Assessment & Explanation Engine `[PLANNED]`
* **Objective:** Ingest extracted clauses from Stage 4 and evaluate their legal risk using category-specific risk rules, scoring matrices, and structured plain-English explanations.
* **Core Components:**
  * `ml/risk/rules.py`: Deterministic legal pattern matching (e.g., liability capped at \$0 or fees paid vs uncapped; immediate termination vs 30-day notice).
  * `ml/risk/scorer.py`: Normalizes risk into 4 tiers: `CRITICAL`, `HIGH`, `MEDIUM`, `LOW`, plus `NEEDS_REVIEW`.
  * `ml/risk/explainer.py`: Synthesizes concise, user-friendly explanations and recommended redline counter-proposals.
* **Inputs:** Stage 4 JSON schema output (`clauses` list with `category`, `clause_text`, `context`).
* **Outputs:** Structured risk payload containing risk score, severity badge, risk reasoning, and suggested modification.
* **Acceptance Criteria:**
  * 100% test coverage over all 9 categories.
  * Deterministic scoring across identical inputs.
  * High-risk provisions correctly flagged with zero false-safe classifications.

### Stage 6: FastAPI Backend & Local PostgreSQL Storage `[PLANNED]`
* **Objective:** Provide a production-grade local REST API and persistence layer for contracts, analysis runs, extracted clauses, and risk scores.
* **Core Components:**
  * Database schema: `contracts`, `document_chunks`, `analysis_runs`, `extracted_clauses`, `risk_assessments`.
  * API endpoints:
    * `POST /api/v1/documents/upload`: Multipart file upload (PDF, DOCX, TXT).
    * `POST /api/v1/analyze/{document_id}`: Triggers Stages 3–5 pipeline.
    * `GET /api/v1/results/{document_id}`: Retrieves structured clause and risk data.
    * `GET /api/v1/health`: Pipeline and database readiness probe.
* **Dependencies:** `fastapi`, `uvicorn`, `sqlalchemy`, `alembic`, `psycopg2-binary` or `asyncpg`.
* **Acceptance Criteria:** Asynchronous processing, complete CRUD operations, automated API test suite passing.

### Stage 7: Interactive Frontend Application `[PLANNED]`
* **Objective:** Modern, responsive, local web UI for uploading contracts, viewing heatmaps, browsing highlighted text, and inspecting risk explanations.
* **Core Components:**
  * Document viewer with synchronized character-level text highlighting.
  * Side-by-side risk cards sorted by severity (`CRITICAL` down to `LOW`).
  * Plain-English explanation drawer with suggested redline counter-proposals.
  * Filter controls for the 9 clause categories.
* **Tech Stack:** React 19, TypeScript, Vite, Tailwind CSS or Modern Component Library.
* **Acceptance Criteria:** Clean rendering of 50+ page contracts, accurate highlight placement without offset drift, fast interaction latencies.

### Stage 8: End-to-End Integration & Regression Testing `[PLANNED]`
* **Objective:** Validate the entire application flow from file upload in the UI to database persistence and risk visualization.
* **Acceptance Criteria:** Full Cypress / Playwright E2E tests passing; end-to-end processing of sample SEC contracts in under 30 seconds on CPU.

### Stage 9: Packaging, Setup Automation & Release Testing `[PLANNED]`
* **Objective:** One-command local startup script, Docker Compose configuration for local PostgreSQL, and complete developer onboarding verification.
* **Acceptance Criteria:** Fresh clone on Windows/Linux runs successfully using automated bootstrap script.

---

## 7. REPOSITORY AND ENVIRONMENT CONTEXT

### 7.1 Verified Directory Structure `[VERIFIED]`
```
clauseguard/
├── backend/
│   ├── app/
│   │   ├── api/                  # API route handlers (empty skeleton)
│   │   ├── core/                 # App configurations (empty skeleton)
│   │   ├── db/                   # Database session and base models (empty skeleton)
│   │   ├── schemas/              # Pydantic request/response schemas (empty skeleton)
│   │   ├── services/             # Background services (empty skeleton)
│   │   └── main.py               # Minimal FastAPI entrypoint
│   └── venv/                     # Active Python 3.13.5 virtual environment
├── docs/
│   ├── category_summary.csv
│   ├── clauseguard_category_mapping.md
│   ├── dataset_exploration_report.md
│   ├── dataset_statistics.json
│   ├── preprocessing_pipeline.md
│   ├── preprocessing_validation_report.json
│   ├── preprocessing_validation_report.md
│   ├── stage4_span_extraction.md
│   └── training_data_preparation_plan.md
├── evaluation/
│   ├── inlegalbert_val_metrics.json
│   ├── roberta_val_metrics.json
│   ├── selected_model_test_metrics.json
│   ├── stage3_model_comparison.json
│   ├── stage3_model_comparison.md
│   ├── stage4_extraction_metrics.json
│   ├── stage4_extraction_report.md
│   └── stage4_val_metrics.json
├── frontend/
│   ├── src/                      # React/TypeScript components
│   ├── package.json              # React 19, TypeScript, Vite
│   ├── tsconfig.json
│   └── vite.config.ts
├── ml/
│   ├── checkpoints/
│   │   ├── inlegalbert/best_checkpoint/
│   │   ├── roberta/best_checkpoint/       # Stage 3 Winning Baseline (FROZEN)
│   │   └── span_extractor/best_checkpoint/# Stage 4 Checkpoint
│   ├── data/
│   │   ├── processed/            # train, val, test Parquet files
│   │   ├── raw/cuad/             # Raw CUAD v1 JSON files
│   │   └── splits/
│   ├── extraction/
│   │   ├── dataset.py            # PretokenizedSpanDataset
│   │   ├── evaluate.py           # Exact & Partial IoU evaluation logic
│   │   ├── evaluate_test.py      # Dual-strategy test set benchmark script
│   │   ├── model.py              # ClauseGuardSpanExtractor architecture
│   │   ├── token_tagging.py      # Subword BIO alignment & decoding
│   │   └── train.py              # Span extractor training engine
│   ├── inference/
│   │   ├── load_model.py         # Stage 3 Classifier loader
│   │   └── pipeline.py           # Unified ClauseGuardPipeline
│   ├── preprocessing/
│   │   ├── chunking.py
│   │   ├── constants.py
│   │   ├── dataset.py
│   │   ├── labeling.py
│   │   └── splitter.py
│   ├── tests/
│   │   ├── test_preprocessing_pipeline.py  # Stage 2 Tests (10/10 PASS)
│   │   ├── test_stage3_training.py         # Stage 3 Tests (10/10 PASS)
│   │   └── test_stage4_extraction.py       # Stage 4 Tests (11/11 PASS)
│   └── training/
│       ├── config.py
│       ├── evaluate.py
│       ├── evaluate_test.py
│       ├── model.py
│       └── train.py
├── .gitignore
├── README.md
└── requirements.txt
```

### 7.2 Environment & Execution Commands `[VERIFIED]`
* **Operating System:** Windows 10/11
* **Python Virtualenv:** `d:\Major Projects\clauseguard\backend\venv\Scripts\python.exe`
* **Python Version:** `Python 3.13.5`

#### Key Operational Commands:
1. **Run Full Automated Test Suite (31 Tests):**
   ```powershell
   backend\venv\Scripts\python.exe -m pytest ml/tests/ -v
   ```
2. **Run Stage 4 Specific Tests (11 Tests):**
   ```powershell
   backend\venv\Scripts\python.exe -m pytest ml/tests/test_stage4_extraction.py -v
   ```
3. **Run Stage 3 Inference Benchmark:**
   ```powershell
   backend\venv\Scripts\python.exe ml/inference/load_model.py
   ```
4. **Run Stage 4 Extraction Training:**
   ```powershell
   backend\venv\Scripts\python.exe ml/extraction/train.py --epochs 5 --batch_size 8 --lr 4e-3
   ```
5. **Run Official Held-Out Test Evaluation:**
   ```powershell
   backend\venv\Scripts\python.exe ml/extraction/evaluate_test.py
   ```

### 7.3 External Artifacts Required Upon Cloning `[VERIFIED]`
Because model checkpoints (`*.bin`, `*.pt`, `ml/checkpoints/`) and raw data are gitignored:
* Raw CUAD data must be present in `ml/data/raw/cuad/CUADv1.json`.
* Parquet chunks exist in `ml/data/processed/`.
* Model weights for `ml/checkpoints/roberta/best_checkpoint/` and `ml/checkpoints/span_extractor/best_checkpoint/` must be obtained from project storage if cloning fresh.

---

## 8. KNOWN ISSUES, PERFORMANCE AUDITS & TECHNICAL DEBT

### 8.1 Stage 4 Extraction Quality Gap `[CRITICAL / IN PROGRESS]`
While all 31 automated unit tests pass, empirical evaluation on the held-out test set revealed severe performance deficiencies:
* **Test Metrics:**
  * Exact-Match Macro F1: **0.0000**
  * Partial-Match Macro F1 (IoU $\ge$ 0.50): **0.0106**
  * Predictions: 4,186 | Gold Annotations: 798
* **Identified Root Causes Under Investigation:**
  1. **Coordinate System Discrepancy:** Potential mismatch between chunk-local character offsets and document-global character offsets during evaluation.
  2. **Class Imbalance in Token Head:** The background class $O$ comprises >97% of tokens. Training only the linear head for 5 epochs leaves token logits dominated by prior distributions.
  3. **Identical Predictions in Assisted vs Independent:** Strategy 1 and Strategy 2 produced nearly identical predictions because Stage 3 gating with $\tau=0.20$ passed nearly all chunks.
  4. **Subword Boundary Fragmentation:** Subwords inside clauses occasionally dip below continuation thresholds, prematurely truncating predicted spans.

### 8.2 Missing Contract Categories in CUAD `[VERIFIED LIMITATION]`
* General *Indemnification*, *Mandatory Arbitration*, and *Confidentiality* cannot be trained from CUAD v1.
* These must be integrated in future phases using external corpora (LEDGAR, CLAUDETTE, SEC Form 8-K).

### 8.3 CPU Latency Trade-Off `[VERIFIED]`
* Training and evaluating on CPU takes ~15–20 minutes for full test runs.
* Batching and frozen backbone evaluation have mitigated latency, but CUDA acceleration should be enabled if GPU resources become available.

---

## 9. INSTRUCTIONS FOR AI ASSISTANTS WORKING ON CLAUSEGUARD

When operating as an AI coding agent on this codebase, you must strictly obey these directives:

1. **Verify Against Code:** Treat this master context as authoritative guidance, but always verify exact function signatures and schemas in the code before modifying files.
2. **Inspect Before Changing:** Read existing implementations, tests, and configurations before writing new code.
3. **Reuse Existing Components:** Never create duplicate chunkers, tokenizers, or model classes. Reuse `ml/preprocessing/constants.py`, `ml/extraction/token_tagging.py`, etc.
4. **Follow Established Decisions:** Respect the 9-category taxonomy, multi-channel BIO architecture, and 3-state supervision masks.
5. **Integrate Incrementally:** Ensure every code edit integrates with existing contracts and leaves the repository in a passing state.
6. **Maintain Backward Compatibility:** Do not break Stage 2 Parquet schemas or Stage 3 inference signatures.
7. **Add Automated Tests:** Every new capability must be accompanied by pytest unit and regression tests in `ml/tests/`.
8. **Preserve Checkpoints & Splits:** Never overwrite `ml/checkpoints/roberta/best_checkpoint/`. Never leak validation or test data into training.
9. **Never Fabricate Metrics:** Report empirical results truthfully. If F1 is low, report it accurately.
10. **Separate Responsibilities:** Keep clause detection, span extraction, risk scoring, and explanation generation as distinct, decoupled modules.
11. **Avoid Unnecessary Abstractions:** Favor clean, maintainable, straightforward Python/PyTorch code over convoluted frameworks.
12. **Focus on Local Execution:** Design all features for local Windows/Linux execution (FastAPI + local PostgreSQL + Vite UI).
13. **Update Documentation:** Keep `docs/` and `evaluation/` markdown files synchronized when implementation contracts change.
14. **Respect Task Scope:** Work exclusively within the stage authorized by the user. Do not jump ahead to Stage 5 or Stage 6 without approval.

---

## 10. NEXT DEVELOPMENT STAGE

### Active Phase: Stage 4 Extraction Performance Audit & Validation Resolution `[IN PROGRESS]`

Before Stage 5 (Risk Assessment) can begin, the Stage 4 span extractor performance gap must be resolved using **validation data only**:

#### Immediate Action Checklist:
1. **Audit Coordinate Systems:** Verify that `evaluate_predicted_spans` in `ml/extraction/evaluate.py` compares gold spans (`global_start`, `global_end`) and predicted spans (`start_char`, `end_char`) in the exact same document coordinate space.
2. **Validation Error Analysis:** Analyze false positives and false negatives on `val_chunks.parquet` to determine whether errors stem from boundary offset math or model underfitting.
3. **Refine BIO Decoding & Thresholds:** Tune `b_threshold`, `span_threshold`, and `min_chars` exclusively on validation data.
4. **Re-train Linear Head with Targeted Hyperparameters:** Evaluate unfreezing top encoder layers or adjusting class weights ($O, B, I$) on the training split.
5. **Freeze & Verify:** Once validation Macro F1 achieves acceptable localization, update documentation and request user approval before advancing to Stage 5.
