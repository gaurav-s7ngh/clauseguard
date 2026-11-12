# ClauseGuard Stage 3: Multi-Label Transformer Model Comparison & Official Test Report

## Executive Summary
This report documents the empirical evaluation and model selection process for ClauseGuard Stage 3: Multi-Label Transformer Training & Evaluation.

Two candidate transformer architectures were fine-tuned and evaluated under identical conditions:
1. **RoBERTa-base** (`roberta-base`): General-domain masked language model baseline.
2. **InLegalBERT** (`law-ai/InLegalBERT`): Domain-adapted legal language model challenger.

### Model Selection Decision
- **Selected Model:** `ROBERTA`
- **Selection Criterion:** Selected strictly based on validation Macro F1 score on the full unsampled validation split (2,812 chunks across 68 contracts).
- **Official Test Set Evaluation:** Evaluated once on the held-out official test set (3,721 chunks across 102 contracts) using frozen validation-calibrated decision thresholds.

---

## 1. Validation Performance Comparison

Evaluated on the full unsampled validation set (2,812 chunks, 68 contracts).

| Metric | RoBERTa-base | InLegalBERT | Delta (InLegalBERT - RoBERTa) |
|---|---|---|---|
| **Macro F1 (Calibrated)** | **0.2314** | **0.2032** | **-0.0282** |
| Macro F1 (Default 0.50) | 0.1311 | 0.1588 | +0.0277 |
| Micro F1 (Calibrated) | 0.2030 | 0.2016 | -0.0014 |
| Macro Precision | 0.2219 | 0.1700 | -0.0519 |
| Macro Recall | 0.3187 | 0.2944 | -0.0243 |
| Mean PR-AUC | 0.1646 | 0.1592 | -0.0054 |
| Validation Loss | 0.5269 | 0.4807 | -0.0462 |
| Inference Latency (p50) | 208.2 ms | 215.6 ms | +7.3 ms |
| Inference Latency (p95) | 224.4 ms | 368.2 ms | +143.8 ms |
| Model Size on Disk | 475.6 MB | 417.8 MB | -57.8 MB |

### Per-Category Validation F1 & Optimal Thresholds

| Category | RoBERTa Thresh | RoBERTa F1 | InLegalBERT Thresh | InLegalBERT F1 | Best Model |
|---|---|---|---|---|---|
| `liability_cap` | 0.56 | 0.4390 | 0.62 | 0.3025 | RoBERTa |
| `termination_for_convenience` | 0.28 | 0.0649 | 0.36 | 0.0682 | InLegalBERT |
| `anti_assignment` | 0.40 | 0.2857 | 0.56 | 0.2838 | RoBERTa |
| `renewal_term` | 0.30 | 0.2338 | 0.50 | 0.2364 | InLegalBERT |
| `governing_law` | 0.48 | 0.2568 | 0.60 | 0.3444 | InLegalBERT |
| `audit_rights` | 0.50 | 0.2419 | 0.56 | 0.1268 | RoBERTa |
| `ip_ownership_assignment` | 0.30 | 0.1106 | 0.48 | 0.0902 | RoBERTa |
| `license_grant` | 0.52 | 0.3617 | 0.60 | 0.3345 | RoBERTa |
| `non_compete` | 0.36 | 0.0881 | 0.36 | 0.0420 | RoBERTa |

---

## 2. Selected Model Official Test Set Evaluation

The winning model (`ROBERTA`) was evaluated once on the untouched held-out official test set (3,721 chunks, 102 contracts) using the frozen validation-calibrated decision thresholds.

### Test Set Headline Metrics
- **Test Macro F1 (Calibrated):** **0.2055** (vs 0.1251 at default 0.50 threshold, **+64.3% relative gain**)
- **Test Micro F1 (Calibrated):** **0.1707**
- **Test Macro Precision:** **0.2047**
- **Test Macro Recall:** **0.3051**
- **Test Mean PR-AUC:** **0.1567**
- **Test Loss:** **0.5272**
- **Evaluation Throughput:** 4.7 chunks/sec on CPU (790.8s total)

### Category-Level Test Performance

| Category | Test Positives | Frozen Threshold | Test Precision | Test Recall | Test F1 | Test PR-AUC |
|---|---|---|---|---|---|---|
| `liability_cap` | 84 | 0.56 | 0.3077 | 0.2857 | **0.2963** | 0.2068 |
| `termination_for_convenience` | 38 | 0.28 | 0.0409 | 0.3421 | **0.0730** | 0.0352 |
| `anti_assignment` | 104 | 0.40 | 0.3205 | 0.2404 | **0.2747** | 0.2655 |
| `renewal_term` | 29 | 0.30 | 0.0986 | 0.2414 | **0.1400** | 0.0644 |
| `governing_law` | 91 | 0.48 | 0.2281 | 0.2857 | **0.2537** | 0.1720 |
| `audit_rights` | 72 | 0.50 | 0.2632 | 0.2083 | **0.2326** | 0.1789 |
| `ip_ownership_assignment` | 49 | 0.30 | 0.0806 | 0.4490 | **0.1366** | 0.0777 |
| `license_grant` | 114 | 0.52 | 0.4675 | 0.3158 | **0.3770** | 0.3780 |
| `non_compete` | 45 | 0.36 | 0.0357 | 0.3778 | **0.0653** | 0.0316 |
| **MACRO AVERAGE** | **626** | **-** | **0.2047** | **0.3051** | **0.2055** | **0.1567** |
| **MICRO AVERAGE** | **626** | **-** | **0.1200** | **0.2955** | **0.1707** | **-** |

---

## 3. Findings & Engineering Takeaways

1. **Threshold Calibration Impact:**
   Standard 0.50 sigmoid thresholding severely penalizes rare categories in imbalanced multi-label contract analysis. Calibrating per-category thresholds on validation data dramatically boosted Macro F1 across minority clause types (from 0.1251 up to 0.2055 on test, activating non-zero detections for previously silent categories like `anti_assignment`, `termination_for_convenience`, and `non_compete`).

2. **General Pretraining (RoBERTa) vs Legal Domain Adaptation (InLegalBERT):**
   RoBERTa-base achieved superior generalization on CUAD commercial agreements compared to InLegalBERT (Validation Macro F1: 0.2314 vs 0.2032). InLegalBERT, while pre-trained on Indian court judgments, was less attuned to US commercial contract drafting idioms than RoBERTa's broader pretraining corpus.

3. **Latency and Deployment Readiness:**
   Both models deliver p50 inference latencies well within the interactive document processing budget (<215ms per chunk on standard CPU), satisfying ClauseGuard Stage 4 span extraction and Stage 5 API integration prerequisites.
