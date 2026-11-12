# ClauseGuard Stage 4: Token-Level Span Extraction & BIO Tagging Architecture

## Executive Summary
This document defines the architectural design, mathematical formulation, offset alignment mechanics, and inference contracts for **ClauseGuard Stage 4: Token-Level Span Extraction & Inference Pipeline**.

Stage 4 extends the chunk-level multi-label classification established in Stage 3 by predicting exact localized clause spans (`char_start`, `char_end`, `clause_text`) across nine target contract categories.

---

## 1. Architectural Design & Multi-Channel BIO Representation

### 1.1 The Multi-Label Overlap Problem
In CUAD and commercial legal contracts, clauses frequently overlap:
- A single contractual paragraph can contain multiple distinct clauses of the same category (e.g., two distinct `license_grant` stipulations).
- Spans belonging to different categories can overlap or share text (e.g., an `ip_ownership_assignment` clause nested inside or overlapping a `license_grant` clause).

In a traditional flat BIO sequence tagging scheme with mutually exclusive labels (e.g., $18 + 1 = 19$ classes: `B-cat`, `I-cat`, `O`), a token can belong to at most one category. Assigning a token to one category silently erases the other.

### 1.2 Nine Independent Category BIO Channels
To model overlapping legal clauses without information loss, ClauseGuard adopts a **Multi-Channel BIO Architecture**:
- For each of the $K = 9$ categories, every token $t_i$ ($i \in \{0, \dots, L-1\}$) is assigned an independent tag:
  $$\hat{y}_{i, k} \in \{O = 0, B = 1, I = 2\}$$
- Each token simultaneously possesses 9 independent 3-way classifications.
- This allows a token to be tagged as `B-license_grant` and `I-ip_ownership_assignment` simultaneously without structural interference.

```
Token Index:          0       1        2           3           4          5          6
Token:              [CLS]  Licensor  hereby     grants     Licensee   exclusive   rights ...
Channel 0 (lic):      O       B        I           I           I          I          I
Channel 1 (ip):       O       O        O           O           B          I          I
...
Channel 8 (comp):     O       O        O           O           O          O          O
```

### 1.3 Supervision Masking
Preserving the Stage 2 data integrity requirement:
- CUAD annotations are verified per contract for specific categories.
- If category $k$ was unverified for contract $C$ (`label_mask[k] == 0`), all tokens for channel $k$ in that chunk are assigned label `-100` (`IGNORE_INDEX`).
- Unknown labels **never** default to negative (`O`) supervision.

---

## 2. Model Architecture & Encoder Reuse

### 2.1 Reusing Stage 3 RoBERTa Backbone
Rather than training a token classifier from scratch or using an un-adapted general model, Stage 4 reuses the fine-tuned RoBERTa encoder weights from Stage 3:
- Checkpoint source: `ml/checkpoints/roberta/best_checkpoint/`
- The encoder parameters already capture legal domain semantics, vocabulary, and clause structure learned during Stage 3 fine-tuning.

### 2.2 Sequence Classification Head
Given token sequence length $L$ and hidden size $H = 768$:
1. **Transformer Encoder:**
   $$\mathbf{H} = \text{Encoder}(\mathbf{X}) \in \mathbb{R}^{B \times L \times 768}$$
2. **Dropout Regularization:**
   $$\tilde{\mathbf{H}} = \text{Dropout}(\mathbf{H}, p=0.2)$$
3. **Multi-Channel Projection:**
   A single linear projection produces logits for all 9 channels simultaneously:
   $$\mathbf{Z} = \tilde{\mathbf{H}} \mathbf{W}^T + \mathbf{b} \in \mathbb{R}^{B \times L \times (9 \times 3)}$$
   Reshaped to:
   $$\mathbf{Z} \in \mathbb{R}^{B \times L \times 9 \times 3}$$

### 2.3 Loss Function with Class Weighting
Tokens tagged with `O` drastically outnumber `B` and `I` tokens (~98% `O` vs 2% `B`/`I`). To prevent collapse to trivial background predictions, the loss applies class weighting:
$$\mathbf{w}_{\text{class}} = [w_O = 1.0, w_B = 8.0, w_I = 4.0]$$

For each category $k$:
$$\mathcal{L}_k = \text{CrossEntropyLoss}\left(\mathbf{Z}_{:, :, k, :}, \mathbf{Y}_{:, :, k}, \text{weight}=\mathbf{w}_{\text{class}}, \text{ignore\_index}=-100\right)$$
The chunk loss is averaged across actively supervised channels:
$$\mathcal{L} = \frac{\sum_{k=0}^{K-1} m_k \mathcal{L}_k}{\max\left(\sum_{k=0}^{K-1} m_k, 1\right)}, \quad m_k = \mathbb{I}(\text{channel } k \text{ has supervised tokens})$$

---

## 3. Subword-to-Character Alignment Mechanics

### 3.1 Tokenizer Offsets
Using Hugging Face's `FastTokenizer` with `return_offsets_mapping=True`:
- Each subword token $t_i$ has character bounds:
  $$\text{offset}(t_i) = (\text{start}_i, \text{end}_i)$$
- Special tokens (`<s>`, `</s>`, padding) have $\text{start}_i = \text{end}_i = 0$, assigned target `-100`.

### 3.2 Label Assignment Algorithm
For an annotated span $S = (\text{cat}_k, s_{\text{start}}, s_{\text{end}})$ within chunk text $T$:
1. A token $t_i$ belongs to $S$ if:
   $$\max(\text{start}_i, s_{\text{start}}) < \min(\text{end}_i, s_{\text{end}})$$
2. The first subword overlapping $S$ receives tag `B` (1).
3. Subsequent subwords overlapping $S$ receive tag `I` (2).
4. Subwords outside any span of category $k$ receive tag `O` (0) if channel $k$ is supervised, or `-100` if unverified.

---

## 4. Span Reconstruction & Chunk Boundary Merging

### 4.1 Local Span Decoding
From predicted probabilities $\mathbf{P} \in \mathbb{R}^{L \times 9 \times 3}$ and predicted tags $\hat{\mathbf{Y}} = \text{argmax}(\mathbf{P}, \text{dim}=-1)$:
- A span begins at token $i$ where $\hat{Y}_{i, k} = B$.
- The span extends through contiguous tokens $j \ge i$ where $\hat{Y}_{j, k} \in \{B, I\}$.
- Local character boundaries:
  $$\text{local\_start} = \text{offset}(t_i)[0], \quad \text{local\_end} = \text{offset}(t_{last})[1]$$
- Span confidence is the mean token confidence:
  $$\text{conf} = \frac{1}{|S|} \sum_{m \in S} P(m, k, \hat{Y}_{m, k})$$
- Extracted text:
  $$\text{clause\_text} = T[\text{local\_start} : \text{local\_end}]$$

### 4.2 Global Offset Mapping & Cross-Chunk Merging
- Global contract coordinates:
  $$\text{global\_start} = \text{chunk\_char\_start} + \text{local\_start}$$
  $$\text{global\_end} = \text{chunk\_char\_start} + \text{local\_end}$$
- Because consecutive chunks share 50–80 token overlap, boundary-crossing clauses or duplicate chunk detections are resolved by merging:
  If two spans $S_1, S_2$ of category $k$ satisfy:
  $$\max(S_1.\text{start}, S_2.\text{start}) \le \min(S_1.\text{end}, S_2.\text{end}) + \delta_{\text{merge}}$$
  (where $\delta_{\text{merge}} = 30$ characters for whitespace/punctuation gaps), they are merged into:
  $$S_{\text{merged}} = (\min(S_1.\text{start}, S_2.\text{start}), \max(S_1.\text{end}, S_2.\text{end}), \max(S_1.\text{conf}, S_2.\text{conf}))$$

---

## 5. Integration Architecture: Stage 3 + Stage 4

### 5.1 Gating Strategies
We evaluate two operational strategies:
1. **Independent Strategy (`independent`):**
   Stage 4 runs token-level span extraction over all chunks and categories regardless of Stage 3 predictions. Highest recall; ensures no clauses are missed due to classification false negatives.
2. **Classifier-Assisted Strategy (`assisted`):**
   Stage 3 multi-label classifier runs first. If category probability $p_k \ge \tau_{\text{assist}}$ (where $\tau_{\text{assist}} = 0.20$, a high-recall candidate threshold), category $k$ is extracted by Stage 4. Chunks with zero candidates are skipped.

The unified inference pipeline provides a clean toggle:
```python
pipeline = ClauseGuardPipeline(
    classifier_dir="ml/checkpoints/roberta/best_checkpoint",
    extractor_dir="ml/checkpoints/span_extractor/best_checkpoint",
    strategy="assisted",  # or "independent"
    assist_threshold=0.20
)
results = pipeline.analyze_document(contract_text, document_id="doc_123")
```

---

## 6. Structured Output Contract for Stage 5

Every extracted clause strictly adheres to the following contract:

```json
{
  "category": "liability_cap",
  "clause_text": "In no event shall either party's aggregate liability exceed $1,000,000.",
  "start_char": 4210,
  "end_char": 4282,
  "confidence": 0.8845,
  "chunk_id": "doc_123_chunk_004",
  "document_id": "doc_123",
  "page_number": null,
  "surrounding_context": "...terms hereof. In no event shall either party's aggregate liability exceed $1,000,000. Neither party shall be liable for indirect...",
  "model_version": "roberta-span-v1",
  "extraction_status": "verified"
}
```

This guarantees seamless downstream ingestion for Stage 5 (Risk Scoring Engine) and Stage 6 (FastAPI / Database integration).
