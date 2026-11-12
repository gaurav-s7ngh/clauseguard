# ClauseGuard | CUAD Dataset Reconnaissance Report

**Project:** ClauseGuard (Academic NLP Legal Document Analyzer)  
**Stage:** Stage 1 — Dataset Reconnaissance & Exploration  
**Author:** NLP / ML Engineering Team  
**Date of Execution:** November 12, 2026  
**Status:** Completed & Verified  

---

## 1. Executive Summary

As part of Stage 1 for the **ClauseGuard** platform, we conducted an end-to-end reconnaissance and empirical evaluation of the **Contract Understanding Atticus Dataset (CUAD) v1**. 

All primary dataset artifacts were programmatically downloaded from official sources, cataloged, checksum-verified, and ingested into `ml/data/raw/cuad/`. A reusable analysis script (`ml/preprocessing/explore_cuad.py`) was developed and executed to compute concrete dataset distributions, validate annotation integrity, test character offset reconstruction, and investigate structural limitations.

### Key Measured Findings

| Dimension | Measured Fact / Statistic | Technical Implication |
| :--- | :--- | :--- |
| **Total Contracts** | 510 commercial contracts | Modest document count requiring rigorous contract-level splitting. |
| **Benchmark Split** | 408 train (80%), 102 test (20%) | Exact 0-contract overlap (zero cross-contract data leakage). |
| **Total Annotation Spans** | 13,823 spans across 41 categories | Substantial span volume, but heavily imbalanced across classes. |
| **Offset Reconstruction** | **13,823 / 13,823 (100.00%) exact character match** | Annotations strictly match raw context slices without offset drift. |
| **Document Length** | Median: 5,006 words (33,143 chars); Max: 47,733 words | Contracts exceed standard 512-token transformer windows by 10x–90x. |
| **Natural Paragraphs** | 74,435 paragraphs across 510 contracts | 97.52% of paragraphs are $\le$ 400 words, fitting neatly within 512 tokens. |
| **Span Containment** | **96.62%** of spans lie fully within a single natural paragraph | Paragraph-level chunking preserves clause boundaries with minimal fragmentation. |
| **Multi-Label Overlap** | 2,122 inter-category span overlaps; 2,084 multi-label paragraphs | Multi-label classification (BCE loss) is mathematically mandatory. |
| **Class Imbalance** | Top category (`Parties`): 2,554 spans; Bottom (`Price Restrictions`): 27 spans | 94x disparity across categories; severe long-tail distribution. |

---

## 2. Dataset Provenance, Inventory & Licensing

### 2.1 Provenance and Authoritative Sources

CUAD was curated and maintained by **The Atticus Project, Inc.**, an alliance of legal and NLP experts, to accelerate open-source research in legal contract review:

- **Original Publication:** *CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review* (Dan Hendrycks, Collin Burns, Anya Chen, Spencer Ball, et al., NeurIPS 2021). [arXiv:2103.06268](https://arxiv.org/abs/2103.06268).
- **Official Website:** [https://www.atticusprojectai.org/cuad/](https://www.atticusprojectai.org/cuad/)
- **Hugging Face Repository:** [https://huggingface.co/datasets/theatticusproject/cuad](https://huggingface.co/datasets/theatticusproject/cuad)
- **GitHub Repository:** [https://github.com/The-Atticus-Project/cuad](https://github.com/The-Atticus-Project/cuad)
- **Underlying Text Corpus:** 510 commercial legal contracts sourced from the public U.S. Securities and Exchange Commission (SEC) EDGAR database across 25 contract types (e.g., Distributor, License, Joint Venture, Supply, Collaboration, and Outsourcing agreements).

### 2.2 Licensing Conditions

- **Dataset License:** Creative Commons Attribution 4.0 International (**CC BY 4.0**).
- **Usage Rights:** Free for both commercial and academic/non-commercial use with proper attribution.
- **Underlying Documents:** The underlying SEC EDGAR contracts are public regulatory filings. The Atticus Project disclaims representations regarding the individual third-party copyright of EDGAR exhibits, but public SEC filings are widely recognized in legal NLP as fair use for research and machine learning evaluation.
- **Compliance Assessment for ClauseGuard:** The CC BY 4.0 license is fully compatible with our academic research objectives and software development.

### 2.3 Downloaded Files & Integrity Verification

All files were downloaded using `ml/preprocessing/download_cuad.py` and stored in `ml/data/raw/cuad/`. Cryptographic hashes and file sizes were recorded in `ml/data/raw/cuad/manifest.json`:

| File Name | Size (Bytes) | SHA256 Checksum | Ingestion Format / Role |
| :--- | :--- | :--- | :--- |
| `CUADv1.json` | 40,128,638 | `ed0b77d85bdf4014d7495800e8e4a70565b48ee6f8a2e5dca9cf8655dbf10eae` | Full SQuAD v2 format (510 contracts) |
| `train_separate_questions.json` | 35,065,844 | `7de21d2bb741ac939e2a839ef9640634a5e0f9f40ea2b82c6833baf7db33dab4` | Official train split (408 contracts, split questions) |
| `test.json` | 7,378,232 | `007b6a40b0c65247f881627375c3d2e9b6aeeb5dfa89957494feed0765a1a073` | Official test split (102 contracts, 41 questions each) |
| `master_clauses.csv` | 3,955,428 | `4da237bec677bf5b02212d523857cd57a801adde60e8021de063c8cc06823720` | Tabular review format (510 rows x 83 columns) |
| `category_descriptions.csv` | 8,987 | `7499950ee04d2ed2841c0f8ec2ef96b91be039bc2ea3c4edcc36334a465b1f36` | 41 categories, questions, and format groups |
| `CUAD_v1_README.txt` | 26,340 | `57210acaea19734264090aa18ebf7b5c26dcf67796d61cd6eb2b068ebc1e7737` | Atticus Project dataset documentation and datasheet |

---

## 3. Dataset Structure & Annotation Schema

### 3.1 Raw JSON Hierarchy (SQuAD v2 Format)

In `CUADv1.json`, the dataset adopts the Stanford Question Answering Dataset (SQuAD v2.0) schema, adapted for legal clause extraction:

```json
{
  "version": "v1.0",
  "data": [
    {
      "title": "<CONTRACT_NAME>",
      "paragraphs": [
        {
          "context": "<FULL_UNTRUNCATED_CONTRACT_TEXT>",
          "qas": [
            {
              "id": "<CONTRACT_NAME>__<CATEGORY_NAME>",
              "question": "Highlight the parts (if any) of this contract related to \"<Category>\" that should be reviewed by a lawyer. Details: <Description>",
              "is_impossible": false,
              "answers": [
                {
                  "text": "<EXACT_CLAUSE_TEXT>",
                  "answer_start": 12845
                }
              ]
            }
          ]
        }
      ]
    }
  ]
}
```

### 3.2 Critical Structural Observations

1. **Monolithic Paragraph per Document:** Unlike standard SQuAD (where a document is divided into short 100–300 word Wikipedia paragraphs), CUAD places the **entire contract** into a single `context` string under `paragraphs[0]`.
2. **Multi-Span Answers:** While standard SQuAD queries typically expect a single answer span, legal clauses often recur throughout a contract (e.g., party definitions in recitals and signature blocks, multiple termination provisions). In `CUADv1.json`, all matching spans for a category are grouped in the single `answers` list.
3. **Train Question Splitting:** In `train_separate_questions.json`, the authors duplicated queries with index suffixes (e.g., `...__Parties_0`, `...__Parties_1`) so standard single-span SQuAD QA models could be trained. In `test.json`, all spans remain grouped under 41 category questions for multi-span evaluation.
4. **Negative Queries (`is_impossible: True`):** When a contract does not contain a given clause category, `answers` is empty `[]` and `is_impossible` is set to `True`. Across 20,910 total contract-category queries, **14,208 (67.95%)** are unannotated negative queries.

---

## 4. Empirical Dataset Statistics

All statistics in this section were calculated directly from the raw files by `ml/preprocessing/explore_cuad.py` and exported to `docs/dataset_statistics.json`.

### 4.1 Global High-Level Counts

- **Total Documents:** 510 contracts
- **Official Train Documents:** 408 contracts (80.0%)
- **Official Test Documents:** 102 contracts (20.0%)
- **Document Overlap (Train $\cap$ Test):** 0 contracts (strictly disjoint)
- **Total Categories:** 41 legal clause categories
- **Total QA Queries:** 20,910 queries ($510 \times 41$)
- **Positive Queries:** 6,702 (32.05%)
- **Negative Queries:** 14,208 (67.95%)
- **Total Annotation Spans:** 13,823 spans (Train: 11,180; Test: 2,643)
- **Total Natural Paragraphs:** 74,435 paragraphs

### 4.2 Document Length Distributions

Contract documents in CUAD are long-form legal texts with significant variance:

| Metric | Minimum | 25th Pct | Median | Mean | 75th Pct | 90th Pct | Maximum | Std Dev |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Characters** | 645 | 16,416 | 33,143 | 52,563 | 66,395 | 122,117 | 338,211 | 55,892 |
| **Words** | 109 | 2,457 | 5,006 | 7,861 | 10,196 | 18,610 | 47,733 | 8,359 |
| **Paras / Doc** | 1 | 30 | 72 | 146 | 177 | 364 | 2,012 | 204 |

### 4.3 Paragraph Length Distributions

Analyzing natural paragraph boundaries (splitting on double newlines `\n\s*\n` with single newline fallback):

| Metric | Minimum | 25th Pct | Median | Mean | 75th Pct | 90th Pct | Maximum | Std Dev |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Characters** | 1 | 37 | 137 | 355 | 376 | 845 | 9,575 | 638 |
| **Words** | 1 | 5 | 20 | 54 | 57 | 129 | 1,426 | 99 |

- **Paragraphs exceeding 400 words:** Only **1,846 out of 74,435 paragraphs (2.48%)**.
- **Implication:** Over **97.5%** of natural paragraphs naturally fit within a standard 512-token transformer window.

---

## 5. Offset Reconstruction & Validation Results

To verify dataset integrity and eliminate the possibility of corrupted or synthetic data, our exploration script performed an exhaustive check on every annotation span across all 510 contracts:

$$\text{slice} = \text{context}[\text{answer\_start} : \text{answer\_start} + \text{len}(\text{text})]$$

### Validation Outcome:
- **Total Annotations Checked:** 13,823
- **Exact Matches ($\text{slice} == \text{text}$):** **13,823 (100.00%)**
- **Mismatches:** **0 (0.00%)**
- **Span Boundary Alignment:**
  - **Spans fully inside a single natural paragraph:** 13,356 (**96.62%**)
  - **Spans crossing paragraph boundaries:** 467 (**3.38%**)

> [!NOTE]
> The 3.38% of spans crossing paragraph breaks occur because SEC EDGAR filings frequently contain arbitrary ASCII column-formatting line breaks (e.g. `\n\n\n` between words or indented numbered lists). A paragraph chunker must merge small fragmented lines to preserve full sentences.

---

## 6. Comprehensive Distribution Across All 41 CUAD Categories

The table below catalogs all 41 categories in CUAD, ranked by total annotation span frequency. All numbers reflect exact computed values from `docs/dataset_statistics.json` and `docs/category_summary.csv`.

| # | Original CUAD Category | Atticus Group | Answer Format | Positive Contracts | Contract Coverage (%) | Total Spans | Train Spans | Test Spans | Avg Spans / Pos Contract | Median Words | Max Words |
| :-: | :--- | :-: | :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| 1 | **Parties** | — | Entity names | 509 | 99.8% | 2,554 | 2,058 | 496 | 5.02 | 2 | 88 |
| 2 | **License Grant** | 4 | Yes/No | 255 | 50.0% | 777 | 647 | 130 | 3.05 | 53 | 394 |
| 3 | **Cap on Liability** | 6 | Yes/No | 275 | 53.9% | 672 | 535 | 137 | 2.44 | 49 | 299 |
| 4 | **Anti-Assignment** | 3 | Yes/No | 374 | 73.3% | 654 | 532 | 122 | 1.75 | 35 | 375 |
| 5 | **Audit Rights** | 5 | Yes/No | 214 | 42.0% | 643 | 516 | 127 | 3.00 | 40 | 288 |
| 6 | **Insurance** | — | Yes/No | 166 | 32.5% | 560 | 437 | 123 | 3.37 | 37 | 350 |
| 7 | **Document Name** | — | Contract Name | 510 | 100.0% | 521 | 417 | 104 | 1.02 | 3 | 13 |
| 8 | **Agreement Date** | 1 | Date | 470 | 92.2% | 476 | 378 | 98 | 1.01 | 3 | 18 |
| 9 | **Expiration Date** | 1 | Date / Perpetual | 413 | 81.0% | 467 | 382 | 85 | 1.13 | 31 | 247 |
| 10 | **Governing Law** | — | Jurisdiction | 437 | 85.7% | 464 | 374 | 90 | 1.06 | 29 | 141 |
| 11 | **Post-Termination Services** | 5 | Yes/No | 182 | 35.7% | 450 | 363 | 87 | 2.47 | 56 | 275 |
| 12 | **Effective Date** | 1 | Date | 390 | 76.5% | 447 | 359 | 88 | 1.15 | 5 | 117 |
| 13 | **Minimum Commitment** | — | Yes/No | 165 | 32.4% | 424 | 338 | 86 | 2.57 | 43 | 331 |
| 14 | **Revenue/Profit Sharing** | — | Yes/No | 166 | 32.5% | 418 | 344 | 74 | 2.52 | 44 | 325 |
| 15 | **Exclusivity** | 2 | Yes/No | 180 | 35.3% | 410 | 345 | 65 | 2.28 | 46 | 328 |
| 16 | **Rofr/Rofo/Rofn** | — | Yes/No | 85 | 16.7% | 367 | 309 | 58 | 4.32 | 52 | 373 |
| 17 | **IP Ownership Assignment** | — | Yes/No | 124 | 24.3% | 318 | 266 | 52 | 2.56 | 52 | 300 |
| 18 | **Non-Transferable License** | 4 | Yes/No | 138 | 27.1% | 298 | 239 | 59 | 2.16 | 51 | 326 |
| 19 | **Non-Compete** | 2 | Yes/No | 119 | 23.3% | 259 | 206 | 53 | 2.18 | 48 | 479 |
| 20 | **Change of Control** | 3 | Yes/No | 121 | 23.7% | 253 | 206 | 47 | 2.09 | 49 | 326 |
| 21 | **Termination for Convenience** | — | Yes/No | 183 | 35.9% | 246 | 196 | 50 | 1.34 | 27 | 333 |
| 22 | **Renewal Term** | 1 | Successive term | 176 | 34.5% | 210 | 167 | 43 | 1.19 | 44 | 461 |
| 23 | **Warranty Duration** | — | Duration | 75 | 14.7% | 176 | 147 | 29 | 2.35 | 40 | 334 |
| 24 | **Covenant Not to Sue** | — | Yes/No | 100 | 19.6% | 173 | 129 | 44 | 1.73 | 48 | 340 |
| 25 | **Volume Restriction** | — | Yes/No | 82 | 16.1% | 171 | 145 | 26 | 2.09 | 37 | 161 |
| 26 | **Uncapped Liability** | 6 | Yes/No | 111 | 21.8% | 167 | 134 | 33 | 1.50 | 65 | 262 |
| 27 | **Irrevocable or Perpetual License** | 4 | Yes/No | 70 | 13.7% | 165 | 129 | 36 | 2.36 | 71 | 394 |
| 28 | **Competitive Restriction Exception** | 2 | Yes/No | 76 | 14.9% | 125 | 102 | 23 | 1.64 | 55 | 305 |
| 29 | **Notice Period to Terminate Renewal** | 1 | Period | 111 | 21.8% | 122 | 96 | 26 | 1.10 | 46 | 156 |
| 30 | **Liquidated Damages** | — | Yes/No | 61 | 12.0% | 121 | 102 | 19 | 1.98 | 49 | 222 |
| 31 | **Affiliate License-Licensee** | 4 | Yes/No | 59 | 11.6% | 115 | 86 | 29 | 1.95 | 74 | 394 |
| 32 | **Joint IP Ownership** | — | Yes/No | 46 | 9.0% | 115 | 93 | 22 | 2.50 | 46 | 213 |
| 33 | **No-Solicit of Employees** | — | Yes/No | 59 | 11.6% | 91 | 75 | 16 | 1.54 | 56 | 219 |
| 34 | **Affiliate License-Licensor** | 4 | Yes/No | 23 | 4.5% | 69 | 59 | 10 | 3.00 | 80 | 394 |
| 35 | **Source Code Escrow** | — | Yes/No | 13 | 2.5% | 66 | 45 | 21 | 5.08 | 38 | 451 |
| 36 | **Non-Disparagement** | — | Yes/No | 38 | 7.5% | 65 | 50 | 15 | 1.71 | 35 | 292 |
| 37 | **No-Solicit of Customers** | 2 | Yes/No | 34 | 6.7% | 58 | 51 | 7 | 1.71 | 37 | 178 |
| 38 | **Third Party Beneficiary** | — | Yes/No | 32 | 6.3% | 39 | 34 | 5 | 1.22 | 33 | 126 |
| 39 | **Most Favored Nation** | — | Yes/No | 28 | 5.5% | 38 | 33 | 5 | 1.36 | 61 | 176 |
| 40 | **Unlimited/All-You-Can-Eat-License** | 4 | Yes/No | 17 | 3.3% | 32 | 24 | 8 | 1.88 | 48 | 163 |
| 41 | **Price Restrictions** | — | Yes/No | 15 | 2.9% | 27 | 20 | 7 | 1.80 | 46 | 168 |

---

## 7. Multi-Label Overlap & Semantic Co-occurrence

In commercial contracts, legal provisions rarely exist in isolation. A single paragraph frequently addresses multiple intertwined legal concepts.

### 7.1 Quantitative Overlap Metrics

- **Intra-Category Overlapping Spans:** 352 (multiple annotated spans within the same category intersecting or nesting).
- **Inter-Category Overlapping Spans:** 2,122 (spans of different categories intersecting in character offsets).
- **Paragraphs with Multiple Categories:** **2,084 paragraphs** across the dataset contain 2 or more distinct categories.

### 7.2 Top 15 Inter-Category Overlap Pairs

The script identified the most frequent category pairs co-occurring in the exact same text span:

```text
1. Agreement Date                 <---> Effective Date                     (265 overlaps)
2. License Grant                  <---> Non-Transferable License           (216 overlaps)
3. Irrevocable or Perpetual Lic.  <---> License Grant                      (155 overlaps)
4. Cap on Liability               <---> Uncapped Liability                 (141 overlaps)
5. Exclusivity                    <---> License Grant                      (113 overlaps)
6. Notice to Terminate Renewal    <---> Renewal Term                       (105 overlaps)
7. Affiliate License-Licensee     <---> License Grant                      (74 overlaps)
8. Effective Date                 <---> Expiration Date                    (61 overlaps)
9. Affiliate License-Licensor     <---> License Grant                      (48 overlaps)
10. Expiration Date               <---> Renewal Term                       (46 overlaps)
11. Affiliate License-Licensee     <---> Irrevocable or Perpetual License   (37 overlaps)
12. Irrevocable or Perpetual Lic. <---> Non-Transferable License           (35 overlaps)
13. Anti-Assignment               <---> Change of Control                  (34 overlaps)
14. Exclusivity                   <---> Non-Transferable License           (26 overlaps)
15. Competitive Restr. Exception  <---> Exclusivity                        (26 overlaps)
```

### 7.3 Semantic Rationale Behind Major Overlaps

1. **Licensing Cluster:** Drafting conventions routinely combine scope, exclusivity, transferability, and duration in a single sentence:
   > *"Licensor hereby grants to Licensee a non-exclusive, non-transferable, perpetual, irrevocable license..."*
   This simultaneously triggers `License Grant`, `Non-Transferable License`, and `Irrevocable or Perpetual License`.
2. **Liability and Carveouts:** A limitation of liability paragraph almost invariably states the monetary cap and immediately follows with carveouts:
   > *"Except for breaches of Section 8 (Confidentiality) or indemnification obligations [Uncapped Liability], neither party's aggregate liability shall exceed the fees paid in the preceding twelve months [Cap on Liability]."*
3. **Term & Renewal:** Clauses combining duration with auto-renewal trigger `Expiration Date`, `Renewal Term`, and `Notice Period to Terminate Renewal` within a single section.

---

## 8. Representative Annotation Examples

Below are representative excerpts from CUAD verifying how annotators delineated clauses:

### Example 1: Cap on Liability & Uncapped Liability
- **Contract:** `LIMEENERGYCO_09_09_1999-EX-10-DISTRIBUTOR AGREEMENT`
- **Text:**
  > *"IN NO EVENT SHALL EITHER PARTY BE LIABLE TO THE OTHER FOR ANY SPECIAL, CONSEQUENTIAL, INCIDENTAL OR INDIRECT DAMAGES... IN NO EVENT SHALL DISTRIBUTOR'S TOTAL LIABILITY ARISING OUT OF OR RELATED TO THIS AGREEMENT EXCEED THE AMOUNTS PAID TO DISTRIBUTOR HEREUNDER."*
- **Offsets:** Start: 28,142; End: 28,416. Exact character match verified.

### Example 2: Termination for Convenience
- **Contract:** `AmericanWholesaleJewelry_20041215_10QSB_EX-10.1_237583_EX-10.1_Affiliate Agreement`
- **Text:**
  > *"Either party may terminate this Agreement at any time, with or without cause, upon thirty (30) days prior written notice to the other party."*
- **Annotation:** Fully contained within a single 28-word sentence.

### Example 3: Anti-Assignment & Change of Control
- **Contract:** `CreditcardscomInc_20070810_S-1_EX-10.33_362297_EX-10.33_Affiliate Agreement`
- **Text:**
  > *"Neither this Agreement nor any of the rights, interests or obligations hereunder shall be assigned by any of the parties hereto without the prior written consent of the other parties; provided, however, that either party may assign this Agreement in connection with a merger, acquisition, or sale of all or substantially all of its assets."*
- **Annotation:** Triggers both `Anti-Assignment` and `Change of Control`.

---

## 9. Limitations of CUAD's Original SQuAD v2 Format

While CUAD is an extraordinary contribution to legal NLP, casting commercial contract review into SQuAD v2 extractive question answering creates severe practical bottlenecks for a real-time production system:

1. **Computational Infeasibility of Naive Inference:**
   - In SQuAD format, answering 41 categories across a 50-page contract requires sliding a 512-token window over ~100 chunks.
   - Running 41 question queries $\times$ 100 chunks = **4,100 transformer forward passes per contract**.
   - On a CPU or modest GPU, analyzing one contract would take 5 to 15 minutes, which is unacceptable for a web application.
2. **Artificial Question Redundancy:**
   - The question string (`"Highlight the parts (if any) of this contract related to..."`) consumes 25–40 tokens of the precious 512-token input budget in every window, leaving less room for legal document context.
3. **Single vs. Multi-Span Mismatch:**
   - Standard reading comprehension architectures (e.g. SQuAD heads predicting single $P_{start}$ and $P_{end}$ pointers) struggle when a document contains 5 separate occurrences of a clause (such as multiple party names or termination conditions).
4. **Context Truncation at Document Level:**
   - By structuring 50,000-word documents as a single paragraph, standard datasets tools fail out-of-the-box unless custom sliding-window tokenizers are implemented.

### The Solution for ClauseGuard:
Transform the task into a **Hierarchical Two-Stage Pipeline**:
- **Stage A (Filtering / Classification):** A multi-label document/paragraph classifier scans contract chunks in a single pass (100 forward passes instead of 4,100) to identify candidate paragraphs containing target categories.
- **Stage B (Highlighting / Extraction):** Span extraction heads or boundary token classifiers are invoked **only** on positive chunks for the specific detected categories.

---

## 10. Summary of Architectural Recommendations for Stage 2

1. **Category Selection:** Restrict the initial MVP to **9 high-coverage, high-risk categories** (see `docs/clauseguard_category_mapping.md`).
2. **Unit of Preprocessing:** Use **natural paragraph chunking** with sentence-level overflow handling (targeting 250–350 tokens per chunk).
3. **Loss Function:** Train with **Binary Cross-Entropy (`BCEWithLogitsLoss`)** for independent multi-label sigmoid predictions.
4. **Data Splitting:** Maintain strict **contract-level splitting** based on the official 408/102 partition, with zero document leakage.
5. **Handling Dataset Gaps:** Explicitly document that high-priority consumer concepts like **Arbitration**, **Confidentiality**, and **General Indemnification** do not exist in CUAD v1 and require supplemental corpora.
