"""
ClauseGuard Span Extraction Evaluation Engine
=============================================
Evaluates token-level span extraction models on legal contracts:
- Exact-Match evaluation (identical start_char, end_char, and category).
- Partial-Overlap evaluation (character IoU >= 0.50 and category match).
- Per-category and macro/micro precision, recall, and F1.
- Analysis of boundary accuracy and cross-chunk span merging.
"""

from typing import List, Dict, Any, Tuple, Optional, Union
import numpy as np
import torch
from torch.utils.data import DataLoader

from ml.preprocessing.constants import LABEL_NAMES, NUM_LABELS, IDX_TO_LABEL
from ml.extraction.token_tagging import decode_spans_from_bio_tags, merge_overlapping_clauses

def compute_span_iou(
    start1: int, end1: int,
    start2: int, end2: int
) -> float:
    """Computes character-level Intersection over Union (IoU) of two spans."""
    intersection_start = max(start1, start2)
    intersection_end = min(end1, end2)
    intersection = max(0, intersection_end - intersection_start)

    union_start = min(start1, start2)
    union_end = max(end1, end2)
    union = max(1, union_end - union_start)

    return intersection / union

def evaluate_predicted_spans(
    predicted_spans: List[Dict[str, Any]],
    gold_spans: List[Dict[str, Any]],
    iou_threshold: float = 0.50,
    label_names: List[str] = LABEL_NAMES
) -> Dict[str, Any]:
    """
    Computes Exact Match and Partial Overlap (IoU >= threshold) metrics.

    Args:
        predicted_spans: List of dicts with 'category', 'start_char', 'end_char', 'confidence'.
        gold_spans: List of dicts with 'category', 'global_start', 'global_end'.
        iou_threshold: IoU threshold for partial matching (default 0.50).
        label_names: List of target categories.

    Returns:
        Structured metrics dictionary containing exact and partial P, R, F1.
    """
    # Group by category
    preds_by_cat: Dict[str, List[Dict[str, Any]]] = {cat: [] for cat in label_names}
    for p in predicted_spans:
        if p["category"] in preds_by_cat:
            preds_by_cat[p["category"]].append(p)

    golds_by_cat: Dict[str, List[Dict[str, Any]]] = {cat: [] for cat in label_names}
    for g in gold_spans:
        cat = g.get("category")
        if cat in golds_by_cat:
            golds_by_cat[cat].append(g)

    per_category = {}
    tot_exact_tp, tot_exact_fp, tot_exact_fn = 0, 0, 0
    tot_partial_tp, tot_partial_fp, tot_partial_fn = 0, 0, 0

    for cat in label_names:
        preds = preds_by_cat[cat]
        golds = golds_by_cat[cat]

        n_preds = len(preds)
        n_golds = len(golds)

        # Subgroup by document/contract ID to prevent cross-contract match corruption
        doc_ids = set()
        preds_by_doc: Dict[Any, List[Dict[str, Any]]] = {}
        for p in preds:
            d_id = p.get("document_id") or p.get("contract_id")
            preds_by_doc.setdefault(d_id, []).append(p)
            doc_ids.add(d_id)

        golds_by_doc: Dict[Any, List[Dict[str, Any]]] = {}
        for g in golds:
            d_id = g.get("document_id") or g.get("contract_id")
            golds_by_doc.setdefault(d_id, []).append(g)
            doc_ids.add(d_id)

        exact_tp = 0
        partial_tp = 0

        for d_id in doc_ids:
            d_preds = preds_by_doc.get(d_id, [])
            d_golds = golds_by_doc.get(d_id, [])

            # 1. Exact Match evaluation
            matched_golds_exact = set()
            for p in d_preds:
                p_start, p_end = p["start_char"], p["end_char"]
                for g_idx, g in enumerate(d_golds):
                    if g_idx in matched_golds_exact:
                        continue
                    g_start = g.get("global_start", g.get("start_char"))
                    g_end = g.get("global_end", g.get("end_char"))
                    if p_start == g_start and p_end == g_end:
                        matched_golds_exact.add(g_idx)
                        exact_tp += 1
                        break

            # 2. Partial Overlap (IoU >= threshold) evaluation
            matched_golds_partial = set()
            for p in d_preds:
                p_start, p_end = p["start_char"], p["end_char"]
                best_iou = 0.0
                best_g_idx = -1
                for g_idx, g in enumerate(d_golds):
                    if g_idx in matched_golds_partial:
                        continue
                    g_start = g.get("global_start", g.get("start_char"))
                    g_end = g.get("global_end", g.get("end_char"))
                    iou = compute_span_iou(p_start, p_end, g_start, g_end)
                    if iou >= iou_threshold and iou > best_iou:
                        best_iou = iou
                        best_g_idx = g_idx

                if best_g_idx != -1:
                    matched_golds_partial.add(best_g_idx)
                    partial_tp += 1

        exact_fp = n_preds - exact_tp
        exact_fn = n_golds - exact_tp

        exact_prec = exact_tp / (exact_tp + exact_fp) if (exact_tp + exact_fp) > 0 else 0.0
        exact_rec = exact_tp / (exact_tp + exact_fn) if (exact_tp + exact_fn) > 0 else 0.0
        exact_f1 = (2 * exact_prec * exact_rec) / (exact_prec + exact_rec) if (exact_prec + exact_rec) > 0 else 0.0

        partial_fp = n_preds - partial_tp
        partial_fn = n_golds - partial_tp

        partial_prec = partial_tp / (partial_tp + partial_fp) if (partial_tp + partial_fp) > 0 else 0.0
        partial_rec = partial_tp / (partial_tp + partial_fn) if (partial_tp + partial_fn) > 0 else 0.0
        partial_f1 = (2 * partial_prec * partial_rec) / (partial_prec + partial_rec) if (partial_prec + partial_rec) > 0 else 0.0

        per_category[cat] = {
            "gold_count": n_golds,
            "pred_count": n_preds,
            "exact_match": {
                "tp": exact_tp, "fp": exact_fp, "fn": exact_fn,
                "precision": round(exact_prec, 4),
                "recall": round(exact_rec, 4),
                "f1": round(exact_f1, 4)
            },
            "partial_match_iou50": {
                "tp": partial_tp, "fp": partial_fp, "fn": partial_fn,
                "precision": round(partial_prec, 4),
                "recall": round(partial_rec, 4),
                "f1": round(partial_f1, 4)
            }
        }

        tot_exact_tp += exact_tp
        tot_exact_fp += exact_fp
        tot_exact_fn += exact_fn

        tot_partial_tp += partial_tp
        tot_partial_fp += partial_fp
        tot_partial_fn += partial_fn

    # Macro averages
    macro_exact_f1 = float(np.mean([per_category[cat]["exact_match"]["f1"] for cat in label_names]))
    macro_exact_prec = float(np.mean([per_category[cat]["exact_match"]["precision"] for cat in label_names]))
    macro_exact_rec = float(np.mean([per_category[cat]["exact_match"]["recall"] for cat in label_names]))

    macro_partial_f1 = float(np.mean([per_category[cat]["partial_match_iou50"]["f1"] for cat in label_names]))
    macro_partial_prec = float(np.mean([per_category[cat]["partial_match_iou50"]["precision"] for cat in label_names]))
    macro_partial_rec = float(np.mean([per_category[cat]["partial_match_iou50"]["recall"] for cat in label_names]))

    # Micro averages
    micro_exact_prec = tot_exact_tp / (tot_exact_tp + tot_exact_fp) if (tot_exact_tp + tot_exact_fp) > 0 else 0.0
    micro_exact_rec = tot_exact_tp / (tot_exact_tp + tot_exact_fn) if (tot_exact_tp + tot_exact_fn) > 0 else 0.0
    micro_exact_f1 = (2 * micro_exact_prec * micro_exact_rec) / (micro_exact_prec + micro_exact_rec) if (micro_exact_prec + micro_exact_rec) > 0 else 0.0

    micro_partial_prec = tot_partial_tp / (tot_partial_tp + tot_partial_fp) if (tot_partial_tp + tot_partial_fp) > 0 else 0.0
    micro_partial_rec = tot_partial_tp / (tot_partial_tp + tot_partial_fn) if (tot_partial_tp + tot_partial_fn) > 0 else 0.0
    micro_partial_f1 = (2 * micro_partial_prec * micro_partial_rec) / (micro_partial_prec + micro_partial_rec) if (micro_partial_prec + micro_partial_rec) > 0 else 0.0

    return {
        "total_gold_spans": len(gold_spans),
        "total_predicted_spans": len(predicted_spans),
        "exact_match": {
            "macro_precision": round(macro_exact_prec, 4),
            "macro_recall": round(macro_exact_rec, 4),
            "macro_f1": round(macro_exact_f1, 4),
            "micro_precision": round(micro_exact_prec, 4),
            "micro_recall": round(micro_exact_rec, 4),
            "micro_f1": round(micro_exact_f1, 4),
        },
        "partial_match_iou50": {
            "macro_precision": round(macro_partial_prec, 4),
            "macro_recall": round(macro_partial_rec, 4),
            "macro_f1": round(macro_partial_f1, 4),
            "micro_precision": round(micro_partial_prec, 4),
            "micro_recall": round(micro_partial_rec, 4),
            "micro_f1": round(micro_partial_f1, 4),
        },
        "per_category": per_category
    }

def evaluate_extractor_on_dataloader(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str = "cpu",
    min_confidence: float = 0.20,
    class_weights: Optional[torch.Tensor] = None,
    channel_whitelist: Optional[List[int]] = None,
    b_threshold: float = 0.15,
    span_threshold: float = 0.20,
    verbose: bool = True
) -> Tuple[Dict[str, Any], float]:
    """
    Evaluates extraction model over a DataLoader and collects all predicted spans.

    Returns:
        metrics: Span evaluation metrics against ground truth spans.
        avg_loss: Average token cross-entropy loss.
    """
    model.eval()
    model.to(device)

    all_predicted_spans = []
    all_gold_spans = []
    total_loss = 0.0
    total_batches = len(dataloader)

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader, 1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            logits = model(input_ids, attention_mask, token_type_ids)
            loss = model.compute_loss(logits, labels, class_weights=class_weights)
            total_loss += loss.item()

            probs, tags = model.predict_tags_and_probs(logits)
            probs_np = probs.cpu().numpy()
            tags_np = tags.cpu().numpy()

            offset_mapping = batch.get("offset_mapping")
            chunk_ids = batch.get("chunk_id", ["chunk"] * len(input_ids))
            char_starts = batch.get("char_start", [0] * len(input_ids))
            texts = batch.get("text", [""] * len(input_ids))
            contract_ids = batch.get("contract_id")
            if contract_ids is None:
                contract_ids = [None] * len(input_ids)

            if offset_mapping is not None:
                offsets_np = offset_mapping.numpy()
                for b_i in range(len(input_ids)):
                    spans_b = decode_spans_from_bio_tags(
                        tag_predictions=tags_np[b_i],
                        tag_probabilities=probs_np[b_i],
                        offset_mapping=offsets_np[b_i],
                        chunk_text=texts[b_i],
                        chunk_char_start=int(char_starts[b_i]),
                        chunk_id=chunk_ids[b_i],
                        document_id=contract_ids[b_i],
                        min_confidence=min_confidence,
                        channel_whitelist=channel_whitelist,
                        b_threshold=b_threshold,
                        span_threshold=span_threshold
                    )
                    all_predicted_spans.extend(spans_b)

            if verbose and (batch_idx % 20 == 0 or batch_idx == total_batches):
                print(f"    [Span Eval] Batch {batch_idx}/{total_batches} ({batch_idx/total_batches*100:.0f}%) complete")

    avg_loss = total_loss / max(total_batches, 1)
    return all_predicted_spans, avg_loss
