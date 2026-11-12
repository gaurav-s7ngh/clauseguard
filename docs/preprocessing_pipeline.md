# ClauseGuard | Stage 2: Preprocessing Pipeline Architecture & Specification

**Project:** ClauseGuard (AI-Powered Legal Document Analyzer)  
**Stage:** Stage 2 — Dataset Preprocessing & Training Data Pipeline  
**Author:** NLP / ML Engineering Team  
**Date:** November 12, 2026  
**Status:** Implemented, Tested & Verified  

---

## 1. Pipeline Overview & Design Principles

The Stage 2 preprocessing pipeline transforms raw commercial contract filings from the **Contract Understanding Atticus Dataset (CUAD) v1** into model-ready, high-integrity training, validation, and test datasets.

```
Raw CUAD Corpus (510 SEC EDGAR Contracts)
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. DETERMINISTIC CONTRACT-LEVEL SPLITTER                    │
│    - Train: 340 contracts (66.7%)                           │
│    - Validation: 68 contracts (13.3%)                       │
│    - Official Benchmark Test: 102 contracts (20.0%)         │
│    - Guarantees 0-document intersection (ZERO LEAKAGE)      │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. PARAGRAPH-AWARE LEGAL TEXT CHUNKER                       │
│    - Extracts natural paragraphs (\n\s*\n)                  │
│    - Sentence-splits oversized paragraphs (>360 words)      │
│    - Span lookahead merge (prevents clause truncation)      │
│    - Targets 250–350 tokens (~180–260 words) per chunk      │
│    - 100% exact character offset preservation               │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. 9-CATEGORY MULTI-LABEL PROCESSOR & SUPERVISION MASK      │
│    - Maps raw CUAD annotations to 9 ClauseGuard MVP labels  │
│    - Confirmed Positive (1, mask=1)                         │
│    - Verified Negative (0, mask=1)                          │
│    - Unknown Edge Spillover (0, mask=0)                     │
│    - Local offset projection: text[local_s:local_e] == span │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. TOKENIZER-INDEPENDENT PARQUET EXPORT                     │
│    - ml/data/processed/train_chunks.parquet (13,965 chunks) │
│    - ml/data/processed/val_chunks.parquet   (2,812 chunks)  │
│    - ml/data/processed/test_chunks.parquet  (3,721 chunks)  │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. REUSABLE PYTORCH DATASET (ml/preprocessing/dataset.py)   │
│    - On-the-fly tokenization (RoBERTa-base, InLegalBERT)    │
│    - Dynamic subword token offset mapping preservation      │
│    - Fast document highlight projection                     │
└─────────────────────────────────────────────────────────────┘
```

### Core Engineering Principles:
1. **Zero Text Mutation:** Raw context strings are never stripped, lowered, or altered prior to offset indexing. `context[char_start:char_end] == chunk_text` is verified for 100% of chunks.
2. **Tokenizer Independence:** Chunks are saved as clean text with character offsets. Tokenization occurs dynamically at runtime, allowing seamless evaluation of both `roberta-base` (BPE) and `law-ai/InLegalBERT` (WordPiece) without duplicating storage.
3. **Rigorous Contract Isolation:** No contract, paragraph, or chunk crosses split boundaries.
4. **Explicit 3-State Supervision:** Ambiguous boundary fragments are masked out, preventing noisy negative gradients.

---

## 2. Processing Pipeline Steps

### Step 1: Contract-Level Partitioning (`ml/preprocessing/splitter.py`)
- Official CUAD test contracts (102 agreements) are preserved completely untouched.
- The 408 official training contracts are deterministically partitioned into **340 training contracts** and **68 validation contracts** using a fixed pseudo-random seed (`RANDOM_SEED = 42`).
- Contract titles in each split are disjoint:
  $$\mathcal{D}_{\text{train}} \cap \mathcal{D}_{\text{val}} = \emptyset, \quad \mathcal{D}_{\text{train}} \cap \mathcal{D}_{\text{test}} = \emptyset, \quad \mathcal{D}_{\text{val}} \cap \mathcal{D}_{\text{test}} = \emptyset$$

### Step 2: Natural Paragraph Extraction & Oversized Paragraph Handling (`ml/preprocessing/chunking.py`)
- Standard contracts are parsed into natural paragraphs using regex `\S.*?(?=\n\s*\n|\Z)`.
- Flattened SEC filings (contracts without double line breaks) fall back to single newlines `\S.*?(?=\n|\Z)`.
- **Oversized Paragraph Deconstruction:** Any paragraph exceeding `MAX_WORDS_HARD_LIMIT = 360` words is segmented along sentence boundaries (`[^.!?]+[.!?]+|\S+`) with a 35-word sentence overlap, preventing context severing.

### Step 3: Paragraph Accumulation with Span Lookahead
- Paragraphs are accumulated into chunks targeting `TARGET_MAX_WORDS_PER_CHUNK = 240` words ($\approx 320$ subword tokens).
- **Active Span Lookahead:** If a ground-truth clause annotation starts in the current accumulation and extends into the subsequent paragraph, the chunker absorbs the next paragraph (up to 360 words) to keep the clause 100% intact within a single chunk.
- **Empirical Efficacy:** Across all 4,243 target category annotations, **98.49% (4,179 spans)** are completely contained within a single chunk. The remaining 1.51% (64 boundary spans) are explicitly tracked in both adjacent chunks with localized sub-slice coordinates.

### Step 4: Multi-Label Annotation & Supervision Masking (`ml/preprocessing/labeling.py`)
Each chunk receives:
- Multi-hot binary label vector $\mathbf{y} \in \{0, 1\}^9$.
- Supervision mask vector $\mathbf{m} \in \{0, 1\}^9$.
- Detailed span records with exact local offsets relative to chunk text.

---

## 3. Label Semantics & Supervision Mask Specification

ClauseGuard frames clause identification as **multi-label classification** across 9 independent binary decisions:

$$\mathbf{y} = [y_0, y_1, \dots, y_8] \in \{0, 1\}^9$$
$$\mathbf{m} = [m_0, m_1, \dots, m_8] \in \{0, 1\}^9$$

### Label Indexing:
```python
LABEL_NAMES = [
    "liability_cap",                # Index 0 (CUAD: Cap on Liability)
    "termination_for_convenience",  # Index 1 (CUAD: Termination for Convenience)
    "anti_assignment",              # Index 2 (CUAD: Anti-Assignment)
    "renewal_term",                 # Index 3 (CUAD: Renewal Term)
    "governing_law",                # Index 4 (CUAD: Governing Law)
    "audit_rights",                 # Index 5 (CUAD: Audit Rights)
    "ip_ownership_assignment",      # Index 6 (CUAD: IP Ownership Assignment)
    "license_grant",                # Index 7 (CUAD: License Grant)
    "non_compete",                  # Index 8 (CUAD: Non-Compete)
]
```

### The Three Operational Label States:

| State | $y_k$ | $m_k$ | Condition in CUAD & Chunk | Training Behavior |
| :--- | :---: | :---: | :--- | :--- |
| **Confirmed Positive** | `1` | `1` | Chunk contains a full clause span OR a substantive overlap ($\ge 30$ chars or $\ge 30\%$ of total clause length). | Supervised as positive ($+1$). |
| **Verified Negative** | `0` | `1` | (a) Contract is certified `is_impossible == True` (0 mentions in entire contract), OR (b) chunk has zero overlap with any clause of category $k$. | Supervised as negative ($0$). |
| **Unknown / Ambiguous** | `0` | `0` | Chunk has minimal edge overlap ($< 30$ chars and $< 30\%$ of clause length) spilling over a boundary. | **Masked out ($0$).** Excluded from loss computation to prevent noisy negative gradients. |

### Mathematical Loss Formulation for Stage 3 Training:
$$\mathcal{L}(\mathbf{z}, \mathbf{y}, \mathbf{m}) = \frac{\sum_{k=0}^8 m_k \cdot \text{BCEWithLogitsLoss}(z_k, y_k)}{\sum_{k=0}^8 m_k + \epsilon}$$

---

## 4. Output Parquet Schema

Processed datasets are stored in `ml/data/processed/` using Apache Parquet (`pyarrow` engine):

| Field Name | Type | Description |
| :--- | :--- | :--- |
| `contract_id` | `string` | Title of the parent contract from CUAD. |
| `chunk_id` | `string` | Unique identifier: `{contract_id}__chunk_{index:04d}`. |
| `chunk_index` | `int32` | Sequential chunk index within the contract. |
| `split` | `string` | Dataset split: `"train"`, `"val"`, or `"test"`. |
| `text` | `string` | Exact substring slice from original context: `context[char_start:char_end]`. |
| `char_start` | `int64` | Starting character index in original raw contract text. |
| `char_end` | `int64` | Ending character index in original raw contract text. |
| `word_count` | `int32` | Number of whitespace-separated words in chunk text. |
| `approx_tokens` | `int32` | Estimated subword token count ($\approx 1.35 \times \text{word\_count}$). |
| `labels` | `list<int8>` | Array of length 9 containing binary class targets $\{0, 1\}$. |
| `label_mask` | `list<int8>` | Array of length 9 containing binary supervision masks $\{0, 1\}$. |
| `label_names` | `list<string>`| Canonical category names corresponding to label indices. |
| `spans` | `list<struct>` | List of annotation span records contained in this chunk: |
| ↳ `category` | `string` | Target category name. |
| ↳ `global_start` | `int64` | Global start offset in raw contract. |
| ↳ `global_end` | `int64` | Global end offset in raw contract. |
| ↳ `local_start` | `int64` | Local start offset within `text` ($0 \le \text{local\_start} < \text{local\_end} \le \text{len}(\text{text})$). |
| ↳ `local_end` | `int64` | Local end offset within `text`. |
| ↳ `text` | `string` | Exact text slice matching `text[local_start:local_end]`. |
| ↳ `is_cross_chunk` | `bool` | True if this span extends beyond this chunk's boundaries. |
| `span_categories` | `list<string>`| Unique categories present in this chunk. |
| `is_positive` | `bool` | True if at least one category label is positive (`1`). |

---

## 5. PyTorch Dataset & DataLoader (`ml/preprocessing/dataset.py`)

The pipeline provides `ClauseGuardDataset`, a reusable PyTorch `Dataset`:

### Example Usage:
```python
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from ml.preprocessing.dataset import ClauseGuardDataset, clauseguard_collate_fn

# 1. Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("roberta-base")

# 2. Instantiate dataset
train_dataset = ClauseGuardDataset(
    data_source="ml/data/processed/train_chunks.parquet",
    tokenizer=tokenizer,
    max_length=512
)

# 3. Create DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    collate_fn=clauseguard_collate_fn
)

# 4. Batch inspection
for batch in train_loader:
    input_ids = batch["input_ids"]          # Shape: [16, 512] (LongTensor)
    attention_mask = batch["attention_mask"]# Shape: [16, 512] (LongTensor)
    labels = batch["labels"]                # Shape: [16, 9] (FloatTensor)
    label_mask = batch["label_mask"]        # Shape: [16, 9] (FloatTensor)
    offset_mapping = batch["offset_mapping"]# Shape: [16, 512, 2] (LongTensor)
    break
```

### Document Highlight Projection Helper:
The module includes `project_token_span_to_global()` to project local model token predictions directly to original document characters:
$$\text{global\_start} = \text{chunk\_char\_start} + \text{offset\_mapping}[\text{token\_start}][0]$$
$$\text{global\_end} = \text{chunk\_char\_start} + \text{offset\_mapping}[\text{token\_end}][1]$$

---

## 6. Exact Reproduction Instructions

To reproduce the complete preprocessing pipeline from raw CUAD files:

```powershell
# 1. Activate shared Python environment
.\backend\venv\Scripts\Activate.ps1

# 2. Execute dataset preprocessing pipeline
python ml\preprocessing\prepare_dataset.py

# 3. Execute automated test suite
pytest ml\tests\test_preprocessing_pipeline.py -v
```
