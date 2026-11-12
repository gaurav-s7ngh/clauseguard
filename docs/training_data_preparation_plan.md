# ClauseGuard | Training Data Preparation & Model Feasibility Plan

**Project:** ClauseGuard (Legal Document Analyzer)  
**Stage:** Stage 1 Deliverable — Preparation for Stage 2 Preprocessing & Model Training  
**Author:** NLP / ML Engineering Team  
**Date:** November 12, 2026  
**Status:** Approved Technical Architecture  

---

## 1. Problem Formulation & End-to-End Pipeline Architecture

The primary mission of ClauseGuard is twofold:
1. **Document-Level & Section-Level Risk Identification:** Rapidly detect which potentially concerning legal categories (e.g., liability caps, unilateral termination, restrictive covenants) are present in an uploaded agreement.
2. **Interactive Clause Highlighting:** Pinpoint the exact character spans of flagged clauses in the user-facing React UI, enabling real-time inspection, risk scoring, and simple-English explanations.

### 1.1 The Computational Dilemma of CUAD's SQuAD Format

CUAD originally modeled contract analysis as SQuAD v2 Question Answering, querying the entire contract text with 41 distinct question prompts. In production, this architecture breaks down:
- A median contract (5,006 words) spans ~15–20 chunks of 512 tokens.
- A long contract (47,733 words) spans ~120 chunks.
- Running 9 MVP category queries across 100 chunks requires **900 transformer forward passes per contract**.
- On CPU, this requires 5 to 12 minutes per document; on a single GPU, it requires 30–60 seconds.

### 1.2 Proposed Production Architecture: Two-Stage Hierarchical Pipeline

To achieve sub-second document analysis while retaining pinpoint span highlighting, ClauseGuard implements a **Two-Stage Hierarchical Architecture**:

```
Uploaded Contract (PDF / DOCX / TXT)
               │
               ▼
   [Text Extraction & Cleaning]
               │
               ▼
   [Paragraph-Aware Chunking] ───► Stores Global Character Offsets [C_start, C_end]
               │
               ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE 1: Fast Multi-Label Paragraph Classifier             │
│ (RoBERTa-base or InLegalBERT + Multi-Label Head)           │
│ - Single forward pass per chunk (15-100 passes total)       │
│ - Predicts binary probability vector y ∈ [0, 1]^9           │
└─────────────────────────────────────────────────────────────┘
               │
      [Filter Positive Chunks] (Typically 5%–15% of all chunks)
               │
               ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE 2: Targeted Clause Span Extraction & Highlighting     │
│ (Token BIO Classifier or Span QA Head on Positive Chunks)   │
│ - Runs ONLY on chunks where Stage 1 detected target labels  │
│ - Extracts exact token spans [local_start, local_end]       │
└─────────────────────────────────────────────────────────────┘
               │
               ▼
[Global Offset Projection] ──► doc_start = C_start + local_start
               │
               ▼
   React UI Interactive Clause Highlighting & Risk Score Display
```

### Computational Benefit:
By filtering out negative background chunks in Stage 1, Stage 2 processes only 5–15 chunks rather than 100, **reducing total transformer compute by 85% to 92%**.

---

## 2. Paragraph-Aware Chunking & Boundary Alignment

### 2.1 The Failure of Fixed-Stride Sliding Windows

Standard NLP sliding windows (e.g. 512 tokens with a 128-token stride) slice text mechanically based on token counts. In legal documents:
- Sentences are cleaved mid-clause (e.g. splitting an indemnity exception from the main covenant).
- Offset reconciliation becomes messy due to overlapping token windows.
- Computational overhead increases by $3\times$ due to redundant overlapping tokens.

### 2.2 Paragraph-Aware Chunking Strategy

Our empirical evaluation on CUAD established three critical facts:
1. Across all 510 contracts, there are **74,435 natural paragraphs**.
2. **96.62%** of all 13,823 annotated clause spans are **100% contained within a single natural paragraph**.
3. Only **2.48%** of paragraphs exceed 400 words (the threshold for 512 BPE tokens).

### 2.3 The Paragraph-Aware Chunking Algorithm (for Stage 2)

```python
def chunk_legal_document(text: str, target_max_tokens: int = 384, max_tokens: int = 510):
    """
    1. Identify paragraph boundaries using double newlines r'\n\s*\n'.
       (Fallback to single newlines r'\n' for flattened SEC contracts).
    2. Record exact character start and end offsets for each paragraph: [p_start, p_end].
    3. Accumulate consecutive small paragraphs into a single chunk until
       estimated subword token count reaches target_max_tokens (leaving buffer for [CLS]/[SEP]).
    4. If a single paragraph exceeds max_tokens (> 400 words, occurring in 2.48% of cases):
       - Split paragraph along sentence boundaries (using regex r'(?<=[.!?])\s+').
       - Create sub-chunks with a 50-token overlapping sentence window.
    5. Return list of chunks with immutable character offsets:
       {
           'chunk_id': f"{doc_id}_{idx}",
           'text': chunk_text,
           'char_start': chunk_start_idx,
           'char_end': chunk_end_idx,
           'token_count': num_tokens
       }
    """
```

---

## 3. Global Offset Preservation & Document Highlighting

A core requirement of ClauseGuard is rendering interactive highlights over the original uploaded document.

### 3.1 Offset Coordinate Transformation

1. **Raw Document Space:** The original extracted document text is treated as an immutable byte/character buffer of length $L$.
2. **Chunk Local Space:** A chunk covers character span $[C_{\text{start}}, C_{\text{end}}]$ in the document.
3. **Model Token Space:** The transformer tokenizes chunk text into token offsets:
   $$\text{token}_i \longleftrightarrow [t_{\text{start}}^{(i)}, t_{\text{end}}^{(i)}] \quad \text{relative to chunk text}$$
4. **Reconstruction to Document Highlights:**
   $$\text{Highlight}_{\text{start}} = C_{\text{start}} + t_{\text{start}}^{(i)}$$
   $$\text{Highlight}_{\text{end}} = C_{\text{start}} + t_{\text{end}}^{(j)}$$

### 3.2 Verification Guarantee

In Stage 1 reconnaissance, we executed `ml/preprocessing/explore_cuad.py` across all 13,823 annotations in CUAD. **100.00% of character slices reconstructed the exact annotated string**. 

To preserve this precision in Stage 2:
- Never apply destructive text normalizations (such as lowercase conversion, punctuation stripping, or Unicode ASCII replacement) to the raw context before recording character offsets.
- Fast tokenizers from Hugging Face (`PreTrainedTokenizerFast`) must be initialized with `return_offsets_mapping=True` to preserve exact subword character spans.

---

## 4. Multi-Label Ground Truth Construction

### 4.1 Label Assignment Formulation

Let $\mathcal{C}_m = [C_{\text{start}}, C_{\text{end}}]$ denote the character span of chunk $m$.  
Let $\mathcal{S}_k = [S_{\text{start}}, S_{\text{end}}]$ denote an annotated ground-truth span for category $k \in \{1, \dots, 9\}$.

The binary label $y_{m, k}$ for chunk $m$ and category $k$ is defined by span intersection:

$$y_{m, k} = \begin{cases} 
1 & \text{if } \exists \, \mathcal{S}_k \text{ s.t. } \frac{\text{len}(\mathcal{C}_m \cap \mathcal{S}_k)}{\text{len}(\mathcal{S}_k)} \ge \tau \quad \text{or} \quad \text{len}(\mathcal{C}_m \cap \mathcal{S}_k) \ge 30 \text{ chars} \\
0 & \text{otherwise}
\end{cases}$$

Where threshold $\tau = 0.50$ ensures that a chunk containing at least 50% of a clause (or at least 30 characters of substantive legal language) is labeled positive.

### 4.2 Multi-Label Mathematical Objective

Because our reconnaissance proved that **2,084 paragraphs contain multiple distinct categories simultaneously** (e.g., Liability Cap + Uncapped Liability; License Grant + Anti-Assignment), the classification objective must **not** use Softmax Cross-Entropy.

Instead, the network outputs an unnormalized logit vector $\mathbf{z} \in \mathbb{R}^9$, optimized via **Binary Cross-Entropy with Logits (`BCEWithLogitsLoss`)**:

$$\mathcal{L}_{\text{BCE}}(\mathbf{z}, \mathbf{y}) = - \frac{1}{K} \sum_{k=1}^K \left[ y_k \log \sigma(z_k) + (1 - y_k) \log (1 - \sigma(z_k)) \right]$$

To handle category imbalance, we apply positive class weighting $w_k = \frac{N_{\text{neg}, k}}{N_{\text{pos}, k}}$ via the `pos_weight` parameter in PyTorch:

$$\mathcal{L}_{\text{WeightedBCE}}(\mathbf{z}, \mathbf{y}) = - \frac{1}{K} \sum_{k=1}^K \left[ w_k y_k \log \sigma(z_k) + (1 - y_k) \log (1 - \sigma(z_k)) \right]$$

---

## 5. Negative Sampling & Class Imbalance Management

### 5.1 The Sparsity Challenge

Across 74,435 natural paragraphs in CUAD:
- Chunks positive for any of the 9 MVP categories comprise $\sim 7,000$ chunks ($\approx 9.4\%$).
- Chunks negative for all 9 categories comprise $\sim 67,400$ chunks ($\approx 90.6\%$).

Training directly on all 67,400 negative chunks would cause gradient updates to be dominated by negative non-clause signals, suppressing positive recall.

### 5.2 Negative Sampling Strategy for Stage 2

We implement a **Stratified 3:1 Negative Sampling Architecture** for the training split:

| Sample Type | Description | Training Set Composition |
| :--- | :--- | :--- |
| **All Positive Chunks** | Any chunk containing at least 1 of the 9 MVP categories. | 100% included ($\approx 5,500$ chunks) |
| **Near-Miss Legal Negatives** | Chunks containing annotations from the *other 32 non-MVP categories* (e.g. Audit, Insurance, Revenue Share, Warranty). | $\approx 4,000$ chunks |
| **Boilerplate Background Negatives** | Subsampled neutral paragraphs (recitals, address blocks, signatures, index pages). | $\approx 12,000$ chunks |

### 5.3 Strict Evaluation Integrity Rule
- **Training Set:** Uses the controlled 3:1 negative sample to stabilize gradient dynamics.
- **Validation & Test Sets:** **NO DOWNSAMPLING**. Evaluated on the complete, unskewed, real-world distribution of paragraphs to measure genuine precision, recall, and false-positive rates.

---

## 6. Contract-Level Train / Validation / Test Splitting

### 6.1 Absolute Prevention of Data Leakage

Randomly shuffling paragraphs into train and test splits causes catastrophic **data leakage**:
- Contracts from the same company or transaction share identical drafting styles, boilerplate preamble, definitions, and party references.
- If paragraphs from Contract A appear in both train and test, the test set measures memorization of document style rather than generalizable legal clause comprehension.

### 6.2 Data Split Specification

We preserve the official Atticus Project benchmark partition and introduce an internal stratified validation split:

```
Total Corpus: 510 SEC Contracts
├── Benchmark Held-Out Test Set: 102 contracts (20.0%) [CUAD Official test.json]
└── Benchmark Development Corpus: 408 contracts (80.0%) [CUAD Official train]
    ├── Training Set:   340 contracts (66.7% of total)
    └── Validation Set:  68 contracts (13.3% of total)
```

### Stratification Criteria for the 340 / 68 Split:
1. **Contract Type Balance:** Proportional allocation across all 25 contract types (Distributor, License, Supply, etc.).
2. **Category Representation:** Ensure rare MVP categories (e.g., `non_compete` with 119 positive contracts) have proportional representation in validation ($\approx 16$ positive contracts in val, $\approx 80$ in train, $\approx 23$ in test).

---

## 7. Model Feasibility & Comparative Evaluation

We evaluate two candidate transformer backbones for Stage 2: **RoBERTa-base** and **InLegalBERT**.

### 7.1 Architectural & Domain Comparison

| Feature | RoBERTa-base (`roberta-base`) | InLegalBERT (`law-ai/InLegalBERT`) |
| :--- | :--- | :--- |
| **Base Architecture** | RoBERTa (Encoder-only, 12 layers, 768-dim, 12 heads) | BERT (Encoder-only, 12 layers, 768-dim, 12 heads) |
| **Parameter Count** | $\approx 125\text{M}$ parameters | $\approx 110\text{M}$ parameters |
| **Pre-training Corpus** | 160 GB general English (BookCorpus, CC-News, OpenWebText, Stories) | Indian legal judgments (Supreme Court & High Courts), Statutes |
| **Vocabulary & Tokenizer** | Byte-Pair Encoding (BPE), 50,265 vocabulary | WordPiece, 30,522 vocabulary (legal domain tailored) |
| **Legal Domain Alignment** | General English; robust legal baseline in CUAD NeurIPS paper | Specialized legal vocabulary, but pre-trained on Indian judicial case law |
| **Contract Language Match** | Good match for US commercial drafting conventions | Partial mismatch: Case law reasoning vs. SEC transactional agreements |
| **Published CUAD Performance** | Official baseline: $\approx 44\%\text{ AUPR}$ on full CUAD | No published benchmark on CUAD in original paper; requires empirical test |
| **Tokenizer Fragmentation** | Higher subword fragmentation on archaic legal terms | Lower subword fragmentation on common legal/statutory terminology |

### 7.2 Hardware Feasibility & Compute Budget

- **Current Environment:** Python 3.13 / PyTorch 2.15.0 on CPU (CUDA unavailable on current host).
- **CPU Training Bottleneck:**
  - Forward + backward pass for 125M parameters on CPU: $\approx 1.8$ seconds per batch (batch size 8).
  - Training on $\approx 15,000$ chunks $\times 3$ epochs $= 45,000$ iterations $\approx 22$ hours on CPU.
- **Feasible Engineering Protocol:**
  1. **Stage 1 & 2 Preprocessing:** Run entirely locally on CPU (tokenization, paragraph chunking, offset mapping, JSON generation take $< 60$ seconds).
  2. **Pipeline Smoke Testing:** Execute 1-epoch / 50-chunk CPU smoke test locally to verify tensor shapes, gradient flow, and loss convergence without errors.
  3. **Full Training Execution:** Train models on GPU instances (e.g. Google Colab T4 / Kaggle / Cloud VM with PyTorch + CUDA), completing 3 epochs in $\approx 15$ minutes per model.
  4. **ONNX Export for Production:** Convert fine-tuned checkpoints to ONNX runtime (already installed in `requirements.txt`: `onnxruntime==1.31.1`), enabling fast CPU inference ($< 50\text{ ms}$ per paragraph) for FastAPI backend integration.

### 7.3 Recommended Experimental Protocol (Stage 2 / Stage 3)

We adhere strictly to empirical verification: **no model will be declared superior without side-by-side experimental evaluation on our held-out test split**.

- **Experiment 1 (Baseline):** `roberta-base` fine-tuned on the 340-contract train split with Weighted BCE loss.
- **Experiment 2 (Domain Candidate):** `law-ai/InLegalBERT` fine-tuned under identical hyperparameters, learning rate ($2\times 10^{-5}$), batch size (16), and optimizer (AdamW with linear warmup).
- **Evaluation Metrics:**
  - Macro and Micro $F_1$-score across all 9 MVP categories.
  - Precision-Recall Area Under Curve (PR-AUC) per category.
  - Span Extraction Token $F_1$ and Exact Match (EM) on positive chunks.
  - Latency (inference milliseconds per 512-token chunk on CPU and GPU).

---

## 8. Unavoidable Compromises & Mitigations

| Identified Compromise | Impact | Engineered Mitigation in Stage 2 |
| :--- | :--- | :--- |
| **Cross-Paragraph Spans (3.38%)** | 467 spans in CUAD cross natural `\n\n` paragraph boundaries due to arbitrary SEC column formatting. | Group adjacent fragmented paragraphs ($< 15$ words) into single semantic blocks before chunking. |
| **SEC EDGAR Redactions (`[***]`)** | Companies redact proprietary terms (pricing, royalties) with asterisks or brackets. | Treat redaction tokens as literal text; ensure tokenizers map `***` to subwords without raising decode errors. |
| **Tail Category Exclusion** | 32 of 41 CUAD categories are excluded from Phase 1 MVP. | Focuses capacity on high-precision detection of top commercial risks; provides modular extension path for Stage 3. |
| **Missing Product Concepts** | Indemnification, Arbitration, and Confidentiality are absent from CUAD. | Fully documented as external data requirements; pipeline designed to ingest supplementary LEDGAR/CLAUDETTE chunks. |

---

## 9. Next Steps for Stage 2 Implementation

Upon formal approval of Stage 1 reconnaissance, Stage 2 will execute:
1. Build `ml/preprocessing/chunk_dataset.py` implementing paragraph-aware chunking and offset mapping.
2. Generate immutable, versioned training artifacts:
   - `ml/data/processed/train_chunks.parquet`
   - `ml/data/processed/val_chunks.parquet`
   - `ml/data/processed/test_chunks.parquet`
3. Implement `ml/training/dataset.py` with multi-label PyTorch `Dataset` and `DataLoader`.
4. Validate tensor shapes and run baseline multi-label transformer pipeline.
