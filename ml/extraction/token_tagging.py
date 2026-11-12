"""
ClauseGuard Token-Level Tagging & Span Decoding Engine
======================================================
Core utilities for multi-channel subword-to-character BIO alignment:
- Generates 9-channel BIO label matrices [seq_len, 9] with masked supervision (-100).
- Supports overlapping annotations and multiple clauses per chunk.
- Reconstructs exact localized character spans and original text.
- Merges cross-chunk and overlapping predictions.
- Produces structured output adhering to the ClauseGuard Stage 4 contract.
"""

from typing import List, Dict, Any, Tuple, Optional, Union
import numpy as np
import torch
from transformers import PreTrainedTokenizerFast

from ml.preprocessing.constants import LABEL_NAMES, NUM_LABELS, LABEL_TO_IDX, IDX_TO_LABEL

# BIO tag mappings
TAG_O = 0
TAG_B = 1
TAG_I = 2
NUM_TAGS = 3
IGNORE_INDEX = -100

TAG_NAMES = ["O", "B", "I"]

def align_spans_to_bio_labels(
    text: str,
    spans: List[Dict[str, Any]],
    label_mask: Union[List[float], np.ndarray],
    tokenizer: PreTrainedTokenizerFast,
    max_length: int = 384
) -> Tuple[Dict[str, torch.Tensor], np.ndarray]:
    """
    Tokenizes text and aligns ground-truth spans to 9-channel BIO label matrix.

    Args:
        text: Raw chunk text.
        spans: List of span dictionaries with 'category', 'local_start', 'local_end'.
        label_mask: Binary list/array of shape [9] indicating supervised categories.
        tokenizer: Fast tokenizer returning offsets mapping.
        max_length: Maximum sequence length.

    Returns:
        encoding: Tokenizer output containing input_ids, attention_mask, offset_mapping.
        labels: Tensor of shape [seq_len, 9] with values in {-100, 0, 1, 2}.
    """
    encoding = tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_offsets_mapping=True,
        return_tensors="pt"
    )

    offsets = encoding["offset_mapping"][0].numpy()  # [seq_len, 2]
    seq_len = len(offsets)

    # Initialize label matrix: default to 0 ('O')
    labels = np.zeros((seq_len, NUM_LABELS), dtype=np.int64)

    # Mask special tokens and padding tokens across all channels
    for t_idx, (t_start, t_end) in enumerate(offsets):
        if t_start == t_end:
            labels[t_idx, :] = IGNORE_INDEX

    # For unverified categories (label_mask == 0), set all tokens to IGNORE_INDEX
    for k in range(NUM_LABELS):
        if label_mask[k] == 0:
            labels[:, k] = IGNORE_INDEX

    # Process annotated spans for verified categories
    for span in spans:
        cat_name = span.get("category")
        if cat_name not in LABEL_TO_IDX:
            continue
        cat_idx = LABEL_TO_IDX[cat_name]

        # Skip if category is not supervised
        if label_mask[cat_idx] == 0:
            continue

        s_start = span["local_start"]
        s_end = span["local_end"]

        # Find all tokens overlapping the span: max(t_start, s_start) < min(t_end, s_end)
        overlapping_indices = []
        for t_idx, (t_start, t_end) in enumerate(offsets):
            if t_start == t_end:
                continue
            if max(t_start, s_start) < min(t_end, s_end):
                overlapping_indices.append(t_idx)

        if not overlapping_indices:
            continue

        # First token gets B, subsequent get I
        first_token = overlapping_indices[0]
        labels[first_token, cat_idx] = TAG_B
        for subsequent in overlapping_indices[1:]:
            labels[subsequent, cat_idx] = TAG_I

    return encoding, labels

def decode_spans_from_bio_tags(
    tag_predictions: np.ndarray,
    tag_probabilities: np.ndarray,
    offset_mapping: np.ndarray,
    chunk_text: str,
    chunk_char_start: int,
    chunk_id: str,
    document_id: Optional[str] = None,
    min_confidence: float = 0.20,
    model_version: str = "roberta-span-v1",
    channel_whitelist: Optional[List[int]] = None,
    b_threshold: float = 0.15,
    span_threshold: float = 0.20,
    min_chars: int = 1,
    max_gap_tokens: int = 1
) -> List[Dict[str, Any]]:
    """
    Decodes 9-channel BIO predictions into structured clause spans.

    Args:
        tag_predictions: Array [seq_len, 9] with values in {0, 1, 2}.
        tag_probabilities: Array [seq_len, 9, 3] with softmax probabilities.
        offset_mapping: Array [seq_len, 2] with token character offsets.
        chunk_text: Original text of the chunk.
        chunk_char_start: Global character start of this chunk in the document.
        chunk_id: Unique identifier of the chunk.
        document_id: Unique document identifier.
        min_confidence: Minimum span confidence threshold.
        model_version: Model identifier string.
        channel_whitelist: Optional list of channel indices to decode.
        b_threshold: Minimum probability to consider a token as span start (B).
        span_threshold: Minimum probability to consider a token as part of span (I).
        min_chars: Minimum character length of a valid clause span.
        max_gap_tokens: Maximum number of intermediate non-clause tokens to bridge.

    Returns:
        List of structured clause dictionaries adhering to Section 6 contract.
    """
    seq_len, num_cats = tag_predictions.shape
    extracted_clauses = []

    channels_to_decode = range(num_cats) if channel_whitelist is None else channel_whitelist

    for cat_idx in channels_to_decode:
        cat_name = IDX_TO_LABEL[cat_idx]
        tags = tag_predictions[:, cat_idx]
        probs = tag_probabilities[:, cat_idx, :]  # [seq_len, 3]

        i = 0
        while i < seq_len:
            # Check for span start: tag B, or high B-probability, or IO recovery
            is_start = False
            if offset_mapping[i, 0] != offset_mapping[i, 1]:
                if tags[i] == TAG_B:
                    is_start = True
                elif probs[i, TAG_B] >= b_threshold and probs[i, TAG_B] > probs[i, TAG_I]:
                    is_start = True
                elif tags[i] == TAG_I and (i == 0 or tags[i - 1] == TAG_O):
                    is_start = True

            if is_start:
                span_tokens = [i]
                j = i + 1
                tentative_gap: List[int] = []

                while j < seq_len and offset_mapping[j, 0] != offset_mapping[j, 1]:
                    # If next token is an explicit new B tag, conclude current span
                    if tags[j] == TAG_B:
                        break

                    is_continuation = (
                        tags[j] == TAG_I or
                        probs[j, TAG_I] >= span_threshold or
                        (probs[j, TAG_B] + probs[j, TAG_I] >= span_threshold)
                    )

                    if is_continuation:
                        if tentative_gap:
                            span_tokens.extend(tentative_gap)
                            tentative_gap = []
                        span_tokens.append(j)
                    else:
                        if len(tentative_gap) < max_gap_tokens:
                            tentative_gap.append(j)
                        else:
                            break
                    j += 1

                # Span character bounds relative to chunk
                local_start = int(offset_mapping[span_tokens[0], 0])
                local_end = int(offset_mapping[span_tokens[-1], 1])
                clause_text = chunk_text[local_start:local_end]

                # Token-averaged confidence
                span_conf = float(np.mean([
                    probs[t, tags[t]] if tags[t] in (TAG_B, TAG_I) else (probs[t, TAG_B] + probs[t, TAG_I])
                    for t in span_tokens
                ]))

                if len(clause_text.strip()) >= min_chars and span_conf >= min_confidence and local_end > local_start:
                    global_start = chunk_char_start + local_start
                    global_end = chunk_char_start + local_end

                    # Surrounding context (up to 100 chars before and after)
                    ctx_start = max(0, local_start - 100)
                    ctx_end = min(len(chunk_text), local_end + 100)
                    surrounding = chunk_text[ctx_start:ctx_end]

                    extracted_clauses.append({
                        "category": cat_name,
                        "clause_text": clause_text,
                        "start_char": global_start,
                        "end_char": global_end,
                        "confidence": round(span_conf, 4),
                        "chunk_id": chunk_id,
                        "document_id": document_id,
                        "page_number": None,
                        "surrounding_context": surrounding,
                        "model_version": model_version,
                        "extraction_status": "extracted" if span_conf >= 0.50 else "low_confidence"
                    })

                i = j
            else:
                i += 1

    return extracted_clauses

def merge_overlapping_clauses(
    clauses: List[Dict[str, Any]],
    full_document_text: Optional[str] = None,
    merge_gap_chars: int = 40
) -> List[Dict[str, Any]]:
    """
    Merges duplicate or cross-chunk overlapping predictions belonging to the same category.

    Args:
        clauses: List of extracted clause dictionaries.
        full_document_text: Complete contract text for exact reconstruction if available.
        merge_gap_chars: Maximum character gap between adjacent spans to merge.

    Returns:
        Deduplicated, merged list of clauses sorted by start_char.
    """
    if not clauses:
        return []

    # Group by category
    by_category: Dict[str, List[Dict[str, Any]]] = {}
    for cl in clauses:
        by_category.setdefault(cl["category"], []).append(cl)

    merged_results = []

    for cat_name, cat_clauses in by_category.items():
        # Sort by start_char then end_char
        sorted_clauses = sorted(cat_clauses, key=lambda c: (c["start_char"], c["end_char"]))

        current = sorted_clauses[0]

        for nxt in sorted_clauses[1:]:
            # Check overlap or small gap
            if nxt["start_char"] <= current["end_char"] + merge_gap_chars:
                # Merge into current
                merged_start = min(current["start_char"], nxt["start_char"])
                merged_end = max(current["end_char"], nxt["end_char"])
                merged_conf = max(current["confidence"], nxt["confidence"])

                # Reconstruct merged clause text
                if full_document_text is not None and merged_end <= len(full_document_text):
                    merged_text = full_document_text[merged_start:merged_end]
                else:
                    # If full document text not supplied, splice texts accurately
                    if nxt["end_char"] > current["end_char"]:
                        gap = nxt["start_char"] - current["end_char"]
                        if gap > 0:
                            merged_text = current["clause_text"] + " " + nxt["clause_text"]
                        else:
                            overlap_len = current["end_char"] - nxt["start_char"]
                            overlap_len = max(0, min(overlap_len, len(nxt["clause_text"])))
                            remaining = nxt["clause_text"][overlap_len:]
                            if current["clause_text"].endswith(" ") or remaining.startswith(" "):
                                merged_text = current["clause_text"] + remaining
                            else:
                                merged_text = current["clause_text"] + " " + remaining
                    else:
                        merged_text = current["clause_text"]

                current["start_char"] = merged_start
                current["end_char"] = merged_end
                current["confidence"] = round(merged_conf, 4)
                current["clause_text"] = merged_text
                current["extraction_status"] = "boundary_merged"
            else:
                merged_results.append(current)
                current = nxt

        merged_results.append(current)

    # Sort final merged clauses by global start_char
    merged_results.sort(key=lambda c: (c["start_char"], c["end_char"]))
    return merged_results
