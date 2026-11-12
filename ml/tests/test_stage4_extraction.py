"""
ClauseGuard Stage 4 Token-Level Span Extraction Automated Test Suite
=====================================================================
Validates all 11 mandatory Stage 4 capabilities:
1. Token-to-character alignment.
2. Exact span reconstruction.
3. Multiple clauses in a single chunk.
4. Overlapping categories without conflict.
5. Chunk-boundary crossings and cross-chunk span merging.
6. Long clauses and sequence truncation handling.
7. Masked supervision (unverified categories assigned -100).
8. Duplicate prediction handling / deduplication.
9. Empty / no-clause document processing.
10. Stage 3 -> Stage 4 integrated pipeline execution.
11. Output schema adherence (all 11 required fields).
"""

import sys
import numpy as np
import torch
import pytest
from transformers import AutoTokenizer

from ml.preprocessing.constants import LABEL_NAMES, NUM_LABELS, LABEL_TO_IDX
from ml.extraction.token_tagging import (
    align_spans_to_bio_labels,
    decode_spans_from_bio_tags,
    merge_overlapping_clauses,
    TAG_O, TAG_B, TAG_I, IGNORE_INDEX
)
from ml.extraction.model import ClauseGuardSpanExtractor
from ml.extraction.evaluate import evaluate_predicted_spans
from ml.inference.pipeline import ClauseGuardPipeline

# ============================================================================
# Test 1: Token-to-character alignment
# ============================================================================
def test_token_to_character_alignment():
    """Verify subword token offsets accurately align to original text characters."""
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    text = "The Supplier's total aggregate liability under this Agreement shall not exceed $100,000."
    span = {
        "category": "liability_cap",
        "local_start": 0,
        "local_end": len(text),
        "text": text
    }
    label_mask = [1.0] * NUM_LABELS

    enc, labels = align_spans_to_bio_labels(text, [span], label_mask, tokenizer, max_length=64)
    offsets = enc["offset_mapping"][0].numpy()

    # Find tokens with non-zero character offsets
    valid_tokens = [i for i, (s, e) in enumerate(offsets) if s != e]
    first_tok, last_tok = valid_tokens[0], valid_tokens[-1]

    aligned_text = text[offsets[first_tok][0]:offsets[last_tok][1]]
    assert aligned_text == text, f"Aligned text '{aligned_text}' does not match original '{text}'"


# ============================================================================
# Test 2: Exact span reconstruction
# ============================================================================
def test_exact_span_reconstruction():
    """Verify decoded spans reconstruct the exact substring text and character offsets."""
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    chunk_text = "Preliminary recital text. This Agreement shall be governed by the laws of the State of New York. Concluding recital."
    gold_clause = "This Agreement shall be governed by the laws of the State of New York."
    s_start = chunk_text.find(gold_clause)
    s_end = s_start + len(gold_clause)

    spans = [{
        "category": "governing_law",
        "local_start": s_start,
        "local_end": s_end,
        "text": gold_clause
    }]
    label_mask = [1.0] * NUM_LABELS

    enc, labels = align_spans_to_bio_labels(chunk_text, spans, label_mask, tokenizer, max_length=64)

    # Simulate perfect tag predictions
    tag_preds = np.where(labels != IGNORE_INDEX, labels, TAG_O)
    tag_probs = np.zeros((64, NUM_LABELS, 3))
    tag_probs[:, :, TAG_O] = 0.05
    for i in range(64):
        for k in range(NUM_LABELS):
            tag_probs[i, k, tag_preds[i, k]] = 0.95

    decoded = decode_spans_from_bio_tags(
        tag_predictions=tag_preds,
        tag_probabilities=tag_probs,
        offset_mapping=enc["offset_mapping"][0].numpy(),
        chunk_text=chunk_text,
        chunk_char_start=500,
        chunk_id="chunk_test_01",
        document_id="doc_test_01"
    )

    assert len(decoded) == 1
    assert decoded[0]["category"] == "governing_law"
    assert decoded[0]["clause_text"] == gold_clause
    assert decoded[0]["start_char"] == 500 + s_start
    assert decoded[0]["end_char"] == 500 + s_end

    # Verify evaluation respects document isolation
    eval_match = evaluate_predicted_spans(
        predicted_spans=decoded,
        gold_spans=[{"category": "governing_law", "global_start": 500 + s_start, "global_end": 500 + s_end, "document_id": "doc_test_01"}]
    )
    assert eval_match["per_category"]["governing_law"]["exact_match"]["tp"] == 1

    eval_mismatch = evaluate_predicted_spans(
        predicted_spans=decoded,
        gold_spans=[{"category": "governing_law", "global_start": 500 + s_start, "global_end": 500 + s_end, "document_id": "other_doc"}]
    )
    assert eval_mismatch["per_category"]["governing_law"]["exact_match"]["tp"] == 0
    assert eval_mismatch["per_category"]["governing_law"]["exact_match"]["fp"] == 1
    assert eval_mismatch["per_category"]["governing_law"]["exact_match"]["fn"] == 1


# ============================================================================
# Test 3: Multiple clauses in one chunk
# ============================================================================
def test_multiple_clauses_in_single_chunk():
    """Verify multiple distinct clauses of the same category in one chunk are extracted."""
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    chunk_text = (
        "Licensor grants Licensee a non-exclusive software license. "
        "Separately, Licensor also grants Licensee a documentation license."
    )
    c1 = "Licensor grants Licensee a non-exclusive software license."
    c2 = "Licensor also grants Licensee a documentation license."
    s1_start = chunk_text.find(c1)
    s1_end = s1_start + len(c1)
    s2_start = chunk_text.find(c2)
    s2_end = s2_start + len(c2)

    spans = [
        {"category": "license_grant", "local_start": s1_start, "local_end": s1_end, "text": c1},
        {"category": "license_grant", "local_start": s2_start, "local_end": s2_end, "text": c2}
    ]
    label_mask = [1.0] * NUM_LABELS

    enc, labels = align_spans_to_bio_labels(chunk_text, spans, label_mask, tokenizer, max_length=64)

    tag_preds = np.where(labels != IGNORE_INDEX, labels, TAG_O)
    tag_probs = np.zeros((64, NUM_LABELS, 3))
    tag_probs[:, :, TAG_O] = 0.05
    for i in range(64):
        for k in range(NUM_LABELS):
            tag_probs[i, k, tag_preds[i, k]] = 0.95

    decoded = decode_spans_from_bio_tags(
        tag_predictions=tag_preds,
        tag_probabilities=tag_probs,
        offset_mapping=enc["offset_mapping"][0].numpy(),
        chunk_text=chunk_text,
        chunk_char_start=0,
        chunk_id="chunk_multi"
    )

    assert len(decoded) == 2, f"Expected 2 decoded clauses, got {len(decoded)}"
    assert decoded[0]["clause_text"] == c1
    assert decoded[1]["clause_text"] == c2


# ============================================================================
# Test 4: Overlapping categories without conflict
# ============================================================================
def test_overlapping_categories_multi_channel():
    """Verify two overlapping spans from different categories are both extracted."""
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    chunk_text = "The vendor assigns all intellectual property rights and grants an exclusive commercial license."
    c_ip = "assigns all intellectual property rights and grants an exclusive commercial license"
    c_lic = "grants an exclusive commercial license"

    ip_start = chunk_text.find(c_ip)
    ip_end = ip_start + len(c_ip)
    lic_start = chunk_text.find(c_lic)
    lic_end = lic_start + len(c_lic)

    spans = [
        {"category": "ip_ownership_assignment", "local_start": ip_start, "local_end": ip_end, "text": c_ip},
        {"category": "license_grant", "local_start": lic_start, "local_end": lic_end, "text": c_lic}
    ]
    label_mask = [1.0] * NUM_LABELS

    enc, labels = align_spans_to_bio_labels(chunk_text, spans, label_mask, tokenizer, max_length=64)

    # Verify both channels are simultaneously active
    ip_idx = LABEL_TO_IDX["ip_ownership_assignment"]
    lic_idx = LABEL_TO_IDX["license_grant"]

    assert (labels[:, ip_idx] == TAG_B).sum() == 1
    assert (labels[:, lic_idx] == TAG_B).sum() == 1

    tag_preds = np.where(labels != IGNORE_INDEX, labels, TAG_O)
    tag_probs = np.zeros((64, NUM_LABELS, 3))
    tag_probs[:, :, TAG_O] = 0.05
    for i in range(64):
        for k in range(NUM_LABELS):
            tag_probs[i, k, tag_preds[i, k]] = 0.95

    decoded = decode_spans_from_bio_tags(
        tag_predictions=tag_preds,
        tag_probabilities=tag_probs,
        offset_mapping=enc["offset_mapping"][0].numpy(),
        chunk_text=chunk_text,
        chunk_char_start=0,
        chunk_id="chunk_overlap"
    )

    cats = {d["category"] for d in decoded}
    assert "ip_ownership_assignment" in cats
    assert "license_grant" in cats
    assert len(decoded) == 2


# ============================================================================
# Test 5: Chunk-boundary crossings and merging
# ============================================================================
def test_cross_chunk_boundary_merging():
    """Verify overlapping predictions from consecutive chunks are seamlessly merged."""
    c1 = {
        "category": "non_compete",
        "clause_text": "Employee shall not engage in any competing business for a period of",
        "start_char": 200,
        "end_char": 268,
        "confidence": 0.88,
        "chunk_id": "chunk_01",
        "document_id": "doc_01"
    }
    c2 = {
        "category": "non_compete",
        "clause_text": "competing business for a period of two (2) years following termination.",
        "start_char": 232,
        "end_char": 303,
        "confidence": 0.91,
        "chunk_id": "chunk_02",
        "document_id": "doc_01"
    }

    full_contract = "Preamble. Employee shall not engage in any competing business for a period of two (2) years following termination."
    c1_text = "Employee shall not engage in any competing business for a period of"
    c2_text = "competing business for a period of two (2) years following termination."

    c1["clause_text"] = c1_text
    c1["start_char"] = full_contract.find(c1_text)
    c1["end_char"] = c1["start_char"] + len(c1_text)

    c2["clause_text"] = c2_text
    c2["start_char"] = full_contract.find(c2_text)
    c2["end_char"] = c2["start_char"] + len(c2_text)

    merged = merge_overlapping_clauses([c1, c2], full_document_text=full_contract)
    assert len(merged) == 1
    assert merged[0]["start_char"] == c1["start_char"]
    assert merged[0]["end_char"] == c2["end_char"]
    assert merged[0]["clause_text"] == "Employee shall not engage in any competing business for a period of two (2) years following termination."
    assert merged[0]["extraction_status"] == "boundary_merged"


# ============================================================================
# Test 6: Long clauses and truncation handling
# ============================================================================
def test_long_clause_truncation_safety():
    """Verify that chunks with spans extending past max_length do not crash alignment."""
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    long_text = "word " * 500  # 500 words > 384 tokens
    span = {
        "category": "audit_rights",
        "local_start": 100,
        "local_end": 1800,  # Beyond 64 tokens
        "text": long_text[100:1800]
    }
    label_mask = [1.0] * NUM_LABELS

    enc, labels = align_spans_to_bio_labels(long_text, [span], label_mask, tokenizer, max_length=64)
    assert labels.shape == (64, NUM_LABELS)
    assert labels[-1, LABEL_TO_IDX["audit_rights"]] in [TAG_I, IGNORE_INDEX]


# ============================================================================
# Test 7: Masked supervision
# ============================================================================
def test_masked_supervision_exclusion():
    """Verify unverified categories receive -100 and contribute zero token loss."""
    logits = torch.randn(2, 32, NUM_LABELS, 3)
    labels = torch.zeros(2, 32, NUM_LABELS, dtype=torch.long)

    # Supervise channel 0, mask channels 1..8
    labels[:, :, 1:] = IGNORE_INDEX

    loss_baseline = ClauseGuardSpanExtractor.compute_loss(logits, labels)

    # Perturb masked channels in logits and labels arbitrarily
    logits_mod = logits.clone()
    logits_mod[:, :, 1:, :] += 100.0

    loss_mod = ClauseGuardSpanExtractor.compute_loss(logits_mod, labels)
    assert torch.isclose(loss_baseline, loss_mod, atol=1e-5), "Masked channels must not affect loss"


# ============================================================================
# Test 8: Duplicate prediction handling & document isolation
# ============================================================================
def test_duplicate_prediction_deduplication():
    """Verify duplicate spans from consecutive chunks are deduplicated within same document, but isolated across documents."""
    c1 = {
        "category": "anti_assignment",
        "clause_text": "Neither party may assign this Agreement without prior written consent.",
        "start_char": 400,
        "end_char": 470,
        "confidence": 0.82,
        "chunk_id": "chunk_A",
        "document_id": "doc_1"
    }
    c2 = {
        "category": "anti_assignment",
        "clause_text": "Neither party may assign this Agreement without prior written consent.",
        "start_char": 400,
        "end_char": 470,
        "confidence": 0.94,
        "chunk_id": "chunk_B",
        "document_id": "doc_1"
    }
    c3_other_doc = {
        "category": "anti_assignment",
        "clause_text": "Neither party may assign this Agreement without prior written consent.",
        "start_char": 400,
        "end_char": 470,
        "confidence": 0.90,
        "chunk_id": "chunk_C",
        "document_id": "doc_2"
    }

    # Within same document: deduplicated to highest confidence
    merged_same = merge_overlapping_clauses([c1, c2])
    assert len(merged_same) == 1
    assert merged_same[0]["confidence"] == 0.94
    assert merged_same[0]["start_char"] == 400
    assert merged_same[0].get("document_id") == "doc_1"

    # Across different documents: must NOT merge together despite identical offsets
    merged_cross = merge_overlapping_clauses([c1, c2, c3_other_doc])
    assert len(merged_cross) == 2, "Spans from different documents must remain isolated"
    doc_ids = {m.get("document_id") for m in merged_cross}
    assert doc_ids == {"doc_1", "doc_2"}


# ============================================================================
# Test 9: Empty / no-clause document processing
# ============================================================================
def test_empty_no_clause_document_handling():
    """Verify pipeline gracefully handles clean documents with zero clauses."""
    tag_preds = np.zeros((32, NUM_LABELS), dtype=np.int64)  # All TAG_O
    tag_probs = np.zeros((32, NUM_LABELS, 3))
    tag_probs[:, :, TAG_O] = 0.99

    offsets = np.zeros((32, 2))
    decoded = decode_spans_from_bio_tags(
        tag_predictions=tag_preds,
        tag_probabilities=tag_probs,
        offset_mapping=offsets,
        chunk_text="Just standard boilerplate with no concerning clauses.",
        chunk_char_start=0,
        chunk_id="clean_chunk"
    )
    assert len(decoded) == 0


# ============================================================================
# Test 10: Stage 3 -> Stage 4 integrated pipeline execution
# ============================================================================
def test_integrated_pipeline_execution():
    """Verify ClauseGuardPipeline loads actual checkpoints and runs analysis."""
    pipeline = ClauseGuardPipeline(
        classifier_path="ml/checkpoints/roberta/best_checkpoint",
        extractor_path="ml/checkpoints/span_extractor/best_checkpoint",
        strategy="assisted",
        assist_threshold=0.20
    )

    sample_contract = (
        "This Agreement is between Company A and Company B. "
        "Either party may terminate this Agreement for convenience upon thirty (30) days notice. "
        "This Agreement shall be governed by Delaware law."
    )

    result = pipeline.analyze_document(sample_contract, document_id="test_contract_01")
    assert "document_id" in result
    assert "clauses" in result
    assert "detected_categories" in result
    assert "extracted_categories" in result
    assert result["document_id"] == "test_contract_01"
    assert result["total_chunks"] >= 1


# ============================================================================
# Test 11: Output schema validity
# ============================================================================
def test_output_schema_validity():
    """Verify all extracted clauses strictly contain the 11 required contract fields."""
    required_fields = {
        "category", "clause_text", "start_char", "end_char", "confidence",
        "chunk_id", "document_id", "page_number", "surrounding_context",
        "model_version", "extraction_status"
    }

    dummy_tags = np.zeros((10, NUM_LABELS), dtype=np.int64)
    dummy_tags[2, 0] = TAG_B
    dummy_tags[3, 0] = TAG_I

    dummy_probs = np.zeros((10, NUM_LABELS, 3))
    dummy_probs[2, 0, TAG_B] = 0.85
    dummy_probs[3, 0, TAG_I] = 0.85

    offsets = np.array([[0, 0], [0, 4], [5, 15], [16, 25], [26, 30], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])

    decoded = decode_spans_from_bio_tags(
        tag_predictions=dummy_tags,
        tag_probabilities=dummy_probs,
        offset_mapping=offsets,
        chunk_text="This dummy contract text for schema validation testing.",
        chunk_char_start=100,
        chunk_id="chunk_schema",
        document_id="doc_schema"
    )

    assert len(decoded) == 1
    clause = decoded[0]
    for field in required_fields:
        assert field in clause, f"Missing required contract field: {field}"
    assert isinstance(clause["start_char"], int)
    assert isinstance(clause["end_char"], int)
    assert isinstance(clause["confidence"], float)
    assert isinstance(clause["clause_text"], str)
