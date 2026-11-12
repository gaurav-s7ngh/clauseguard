# ClauseGuard | Stage 2: Preprocessing Validation Report

**Project:** ClauseGuard (AI-Powered Legal Document Analyzer)  
**Document:** Preprocessing Pipeline Validation & Quality Assurance Report  
**Execution Timestamp:** November 12, 2026  
**Pipeline Entry Point:** `ml/preprocessing/prepare_dataset.py`  
**Test Suite:** `ml/tests/test_preprocessing_pipeline.py`  
**Status:** ALL CHECKS PASSED (100% Offset Accuracy, 10/10 Unit Tests Passed)  

---

## 1. Executive Summary

This report documents the verification, empirical counts, and automated quality checks executed for **Stage 2: Dataset Preprocessing & Training Data Pipeline**.

All 510 contracts from CUAD v1 were processed through our deterministic, paragraph-aware legal chunker with active span lookahead. The resulting datasets were validated for character offset reconstruction, label integrity, split isolation, and PyTorch dataset readiness.

### Key Measured Highlights:
- **Output Artifacts Generated:**
  - `ml/data/processed/train_chunks.parquet`: **13,965 chunks** (8.98 MB)
  - `ml/data/processed/val_chunks.parquet`: **2,812 chunks** (1.83 MB)
  - `ml/data/processed/test_chunks.parquet`: **3,721 chunks** (2.41 MB)
  - `docs/preprocessing_validation_report.json`: Machine-readable audit file.
- **Contract-Level Isolation:** 340 train, 68 validation, 102 test contracts. **Zero contract overlap (0 data leakage).**
- **Offset Reconstruction:** **4,243 / 4,243 annotations (100.00%)** verified with exact string equality against raw contract slices. Zero offset mismatches.
- **Validation Exceptions:** **0 errors or malformed annotations.**
- **Test Suite Execution:** **10 out of 10 automated pytest tests PASSED.**

---

## 2. Dataset Split & Chunk Distribution

The table below summarizes document allocation, chunk volume, and class distribution across all three splits:

| Metric | Training Set | Validation Set | Test Set (Official Held-Out) | Total / Combined |
| :--- | :---: | :---: | :---: | :---: |
| **Contract Count** | **340** (66.67%) | **68** (13.33%) | **102** (20.00%) | **510** (100.0%) |
| **Total Chunks** | **13,965** | **2,812** | **3,721** | **20,498** |
| **Positive Chunks ($\ge 1$ label)** | **2,132** | **436** | **575** | **3,143** |
| **Negative Chunks ($0$ labels)** | **11,833** | **2,376** | **3,146** | **17,355** |
| **Positive Chunk Ratio** | **15.27%** | **15.50%** | **15.45%** | **15.33%** |
| **File Size (Parquet)** | 8,976,849 bytes | 1,831,314 bytes | 2,405,829 bytes | 13,213,992 bytes |

> [!NOTE]
> Notice the remarkable consistency of positive chunk ratios across splits: **15.27%** (Train), **15.50%** (Val), and **15.45%** (Test). This demonstrates that our contract-level stratification accurately preserves real-world clause density without skew.

---

## 3. Per-Category Label Distribution Across Splits

For each of the 9 ClauseGuard MVP categories, the pipeline tracks:
- **Positive (`1`):** Chunk contains confirmed substantive clause text.
- **Verified Negative (`0` with mask=1):** Chunk certified absent by CUAD `is_impossible` or non-overlapping paragraph.
- **Unknown (`0` with mask=0):** Minimal boundary spillover ($< 30$ chars & $< 30\%$ clause length) masked out.

### Exact Empirical Counts:

| Category Name | Index | Train (+ / - / ?) | Val (+ / - / ?) | Test (+ / - / ?) | Total Positive Chunks |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`liability_cap`** | 0 | 366 / 13,597 / 2 | 67 / 2,745 / 0 | 84 / 3,637 / 0 | 517 |
| **`termination_for_convenience`**| 1 | 156 / 13,809 / 0 | 25 / 2,787 / 0 | 38 / 3,683 / 0 | 219 |
| **`anti_assignment`** | 2 | 343 / 13,621 / 1 | 68 / 2,744 / 0 | 104 / 3,617 / 0 | 515 |
| **`renewal_term`** | 3 | 135 / 13,830 / 0 | 34 / 2,778 / 0 | 29 / 3,692 / 0 | 198 |
| **`governing_law`** | 4 | 324 / 13,641 / 0 | 53 / 2,759 / 0 | 91 / 3,630 / 0 | 468 |
| **`audit_rights`** | 5 | 290 / 13,675 / 0 | 63 / 2,749 / 0 | 72 / 3,649 / 0 | 425 |
| **`ip_ownership_assignment`** | 6 | 154 / 13,811 / 0 | 37 / 2,775 / 0 | 49 / 3,672 / 0 | 240 |
| **`license_grant`** | 7 | 413 / 13,552 / 0 | 97 / 2,715 / 0 | 114 / 3,607 / 0 | 624 |
| **`non_compete`** | 8 | 133 / 13,830 / 2 | 34 / 2,778 / 0 | 45 / 3,676 / 0 | 212 |

---

## 4. Annotation Integrity & Offset Validation Results

To ensure zero silent data corruption, every annotation was validated against the raw contract text in `ml/data/raw/cuad/CUADv1.json`:

$$\text{slice} = \text{context}[\text{char\_start} : \text{char\_end}]$$
$$\text{local\_slice} = \text{chunk\_text}[\text{local\_start} : \text{local\_end}]$$

### Validation Audit Metrics:
- **Total Annotations Audited:** 4,243
- **Exact String Matches:** 4,243 (**100.00%**)
- **Character Mismatches:** 0 (**0.00%**)
- **Malformed Annotations:** 0
- **Empty Paragraph Exceptions:** 0

### Chunk-to-Context Verification:
For all 20,498 chunks across train, val, and test:
```python
assert contract_context[chunk.char_start : chunk.char_end] == chunk.text
```
**Outcome: 100.00% verified. Zero offset drift.**

---

## 5. Cross-Chunk Annotation Handling

- **Total Cross-Chunk Spans Logged:** 113 chunk-span overlap instances.
- **Handling Strategy:**
  - When an annotation span naturally crosses a chunk boundary (e.g., massive 150-word clauses or indented bullet schedules):
    1. Both Chunk A and Chunk B receive confirmed positive labels if their overlap contains substantive text ($\ge 30$ chars or $\ge 30\%$ of the span).
    2. Local offsets `[local_start, local_end]` are computed relative to each chunk's own text.
    3. The `is_cross_chunk: True` flag is recorded on the span struct.
    4. Global offsets `[global_start, global_end]` preserve the original document coordinates.
- **Result:** Zero spans are truncated or dropped.

---

## 6. Automated Test Suite Execution

We executed the complete test suite in `ml/tests/test_preprocessing_pipeline.py` using `pytest`:

```powershell
backend\venv\Scripts\python.exe -m pytest ml\tests\test_preprocessing_pipeline.py -v
```

### Execution Log:
```text
============================= test session starts =============================
platform win32 -- Python 3.13.5, pytest-9.2.0, pluggy-1.7.0
rootdir: D:\Major Projects\clauseguard
collected 10 items

ml/tests/test_preprocessing_pipeline.py::test_original_annotation_span_reconstruction PASSED [ 10%]
ml/tests/test_preprocessing_pipeline.py::test_global_local_offset_conversion PASSED         [ 20%]
ml/tests/test_preprocessing_pipeline.py::test_label_mapping_across_nine_categories PASSED  [ 30%]
ml/tests/test_preprocessing_pipeline.py::test_positive_negative_unknown_label_semantics PASSED [ 40%]
ml/tests/test_preprocessing_pipeline.py::test_overlapping_and_cross_paragraph_annotations PASSED [ 50%]
ml/tests/test_preprocessing_pipeline.py::test_long_paragraph_splitting_and_chunk_overlap PASSED [ 60%]
ml/tests/test_preprocessing_pipeline.py::test_contract_level_split_isolation PASSED       [ 70%]
ml/tests/test_preprocessing_pipeline.py::test_pytorch_dataset_output_shapes_and_types PASSED [ 80%]
ml/tests/test_preprocessing_pipeline.py::test_reproducibility_with_random_seed PASSED      [ 90%]
ml/tests/test_preprocessing_pipeline.py::test_empty_malformed_invalid_annotation_handling PASSED [100%]

============================= 10 passed in 11.19s =============================
```

### Test Scope Coverage:
1. `test_original_annotation_span_reconstruction`: Verifies `text[local_s:local_e] == span_text` on validation chunks.
2. `test_global_local_offset_conversion`: Verifies $g_{\text{start}} = c_{\text{start}} + l_{\text{start}}$ and `project_token_span_to_global`.
3. `test_label_mapping_across_nine_categories`: Verifies canonical names and indices.
4. `test_positive_negative_unknown_label_semantics`: Verifies 3-state $(y, m)$ combinations.
5. `test_overlapping_and_cross_paragraph_annotations`: Verifies multi-label chunks ($\sum y > 1$) and cross-chunk flags.
6. `test_long_paragraph_splitting_and_chunk_overlap`: Verifies sentence boundary splitting on a 450-word paragraph.
7. `test_contract_level_split_isolation`: Verifies mutual disjointness of 340 train, 68 val, and 102 test contracts.
8. `test_pytorch_dataset_output_shapes_and_types`: Verifies PyTorch tensors `[9]`, batch collation `[B, 9]`, and `roberta-base` tokenized mode `[512]`.
9. `test_reproducibility_with_random_seed`: Verifies seed 42 produces identical splits across runs.
10. `test_empty_malformed_invalid_annotation_handling`: Verifies exception recording on corrupt input without crashes.

---

## 7. Exact Commands to Reproduce Pipeline

All results and files can be reproduced from the project root:

```powershell
# 1. Run the preprocessing pipeline
backend\venv\Scripts\python.exe ml\preprocessing\prepare_dataset.py

# 2. Run the test suite
backend\venv\Scripts\python.exe -m pytest ml\tests\test_preprocessing_pipeline.py -v
```

---

## 8. Definition of Done Compliance Checklist

- [x] All three processed datasets generated (`train_chunks.parquet`, `val_chunks.parquet`, `test_chunks.parquet`).
- [x] All nine labels and supervision masks correctly represented.
- [x] Original annotation offsets reconstructed and validated with 100.00% precision.
- [x] Zero contract leakage between train, validation, and test splits.
- [x] Cross-paragraph and overlapping annotations explicitly handled and preserved.
- [x] Reusable PyTorch `ClauseGuardDataset` implemented, tested, and validated.
- [x] Pipeline deterministic and fully reproducible from documented commands.
- [x] Machine-readable validation report exported (`docs/preprocessing_validation_report.json`).
- [x] Preprocessing architecture and QA reports documented.
