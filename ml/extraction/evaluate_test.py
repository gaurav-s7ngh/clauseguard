"""
ClauseGuard Stage 4 Official Test Set Evaluation & Strategy Comparison
======================================================================
Evaluates the fine-tuned ClauseGuard token-level span extractor on the
untouched official held-out test set (3,721 chunks, 102 contracts).

Compares:
1. Independent Extraction: Evaluates all 9 categories across all chunks.
2. Classifier-Assisted Extraction: Gated by Stage 3 recall-oriented candidate filter (prob >= 0.20).

Outputs:
- evaluation/stage4_extraction_metrics.json
- evaluation/stage4_extraction_report.md
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Add workspace root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ml.preprocessing.constants import LABEL_NAMES, NUM_LABELS, LABEL_TO_IDX
from ml.extraction.token_tagging import decode_spans_from_bio_tags, merge_overlapping_clauses
from ml.extraction.dataset import build_span_dataset
from ml.extraction.model import ClauseGuardSpanExtractor
from ml.extraction.evaluate import evaluate_predicted_spans, evaluate_extractor_on_dataloader
from ml.inference.load_model import ClauseGuardInference

CHECKPOINT_DIR = Path("ml/checkpoints/span_extractor/best_checkpoint")
CLASSIFIER_DIR = Path("ml/checkpoints/roberta/best_checkpoint")
TEST_PARQUET = Path("ml/data/processed/test_chunks.parquet")
EVALUATION_DIR = Path("evaluation")

def run_test_evaluation(
    device: str = "cpu",
    batch_size: int = 32,
    max_length: int = 256,
    assist_threshold: float = 0.20,
    min_confidence: float = 0.20,
    b_threshold: float = 0.15,
    span_threshold: float = 0.20
) -> Dict[str, Any]:
    """
    Runs full test evaluation on all 3,721 test chunks comparing independent and assisted extraction.
    """
    print("=" * 80)
    print("CLAUSEGUARD STAGE 4: OFFICIAL TEST EVALUATION & STRATEGY COMPARISON")
    print(f"Extractor Checkpoint: {CHECKPOINT_DIR}")
    print(f"Classifier Checkpoint: {CLASSIFIER_DIR}")
    print(f"Test Set: {TEST_PARQUET} | Device: {device.upper()}")
    print("=" * 80)

    # 1. Load Extractor and Classifier
    print("\n[1/5] Loading trained span extractor and Stage 3 classifier...")
    extractor, extractor_meta = ClauseGuardSpanExtractor.from_pretrained(CHECKPOINT_DIR, device=device)
    tokenizer = AutoTokenizer.from_pretrained(str(CHECKPOINT_DIR))
    classifier = ClauseGuardInference(CLASSIFIER_DIR, device=device)

    # 2. Load Test Data
    print("\n[2/5] Loading test dataset (3,721 chunks)...")
    cols = ["chunk_id", "contract_id", "text", "labels", "label_mask", "char_start", "char_end", "is_positive", "spans", "span_categories"]
    test_df = pd.read_parquet(TEST_PARQUET, columns=cols)

    all_test_gold_spans = []
    seen_gold = set()
    for _, row in test_df.iterrows():
        c_id = row["contract_id"]
        for s in row["spans"]:
            key = (c_id, s["category"], s["global_start"], s["global_end"])
            if key not in seen_gold:
                seen_gold.add(key)
                g = dict(s)
                g["contract_id"] = c_id
                g["document_id"] = c_id
                all_test_gold_spans.append(g)

    print(f"Loaded {len(test_df):,} test chunks with {len(all_test_gold_spans):,} unique gold clause annotations.")

    test_dataset = build_span_dataset(test_df, tokenizer, max_length=max_length, include_metadata=True)
    del test_df
    import gc; gc.collect()

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    total_batches = len(test_loader)

    # 3 & 4. Dual Strategy Batched Inference
    print(f"\n[3/5] Running dual-strategy batched inference across {len(test_dataset):,} chunks ({total_batches} batches)...")
    indep_raw_spans = []
    assisted_raw_spans = []

    extractor.eval()
    t0_indep = 0.0
    t0_assist = 0.0

    t_start_all = time.time()

    with torch.no_grad():
        for b_idx, batch in enumerate(test_loader, 1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            offsets_np = batch["offset_mapping"].numpy()
            chunk_ids = batch["chunk_id"]
            char_starts = batch["char_start"]
            texts = batch["text"]
            contract_ids = batch.get("contract_id", [None] * b_size)
            b_size = len(input_ids)

            # Extractor forward pass (shared representation)
            t_ext_start = time.time()
            logits = extractor(input_ids, attention_mask)
            probs, tags = extractor.predict_tags_and_probs(logits)
            probs_np = probs.cpu().numpy()
            tags_np = tags.cpu().numpy()
            ext_duration = time.time() - t_ext_start

            # Classifier forward pass for assisted strategy
            t_clf_start = time.time()
            clf_batch = classifier.predict_batch(texts, max_length=256, batch_size=b_size)
            clf_duration = time.time() - t_clf_start

            t0_indep += ext_duration
            t0_assist += (ext_duration + clf_duration)

            for i in range(b_size):
                # 1. Independent decoding (all 9 channels)
                spans_indep = decode_spans_from_bio_tags(
                    tag_predictions=tags_np[i],
                    tag_probabilities=probs_np[i],
                    offset_mapping=offsets_np[i],
                    chunk_text=texts[i],
                    chunk_char_start=int(char_starts[i]),
                    chunk_id=chunk_ids[i],
                    document_id=contract_ids[i],
                    min_confidence=min_confidence,
                    b_threshold=b_threshold,
                    span_threshold=span_threshold
                )
                indep_raw_spans.extend(spans_indep)

                # 2. Assisted decoding (gated channels only using calibrated detection threshold)
                clf_res = clf_batch[i]
                active_channels = [
                    LABEL_TO_IDX[cat] for cat in LABEL_NAMES
                    if clf_res["categories"][cat]["detected"]
                ]
                if active_channels:
                    spans_asst = decode_spans_from_bio_tags(
                        tag_predictions=tags_np[i],
                        tag_probabilities=probs_np[i],
                        offset_mapping=offsets_np[i],
                        chunk_text=texts[i],
                        chunk_char_start=int(char_starts[i]),
                        chunk_id=chunk_ids[i],
                        document_id=contract_ids[i],
                        min_confidence=min_confidence,
                        channel_whitelist=active_channels,
                        b_threshold=b_threshold,
                        span_threshold=span_threshold
                    )
                    assisted_raw_spans.extend(spans_asst)

            if b_idx % 20 == 0 or b_idx == total_batches:
                elapsed = time.time() - t_start_all
                print(f"    Batch {b_idx}/{total_batches} ({b_idx/total_batches*100:.0f}%) complete | Elapsed: {elapsed:.1f}s | Indep Spans: {len(indep_raw_spans):,} | Asst Spans: {len(assisted_raw_spans):,}")

    indep_time = t0_indep
    indep_throughput = len(test_dataset) / max(indep_time, 0.1)

    assist_time = t0_assist
    assist_throughput = len(test_dataset) / max(assist_time, 0.1)

    print(f"\n[4/5] Merging and evaluating predictions...")
    print(f"Merging {len(indep_raw_spans):,} raw spans for Independent Extraction...")
    indep_merged_spans = merge_overlapping_clauses(indep_raw_spans)
    print(f"Deduplicated to {len(indep_merged_spans):,} merged clauses.")

    indep_metrics = evaluate_predicted_spans(
        predicted_spans=indep_merged_spans,
        gold_spans=all_test_gold_spans,
        iou_threshold=0.50,
        label_names=LABEL_NAMES
    )
    indep_metrics["execution_time_seconds"] = round(indep_time, 1)
    indep_metrics["throughput_chunks_per_sec"] = round(indep_throughput, 1)

    print(f"\nMerging {len(assisted_raw_spans):,} raw spans for Classifier-Assisted Extraction...")
    assisted_merged_spans = merge_overlapping_clauses(assisted_raw_spans)
    print(f"Deduplicated to {len(assisted_merged_spans):,} merged clauses.")

    assisted_metrics = evaluate_predicted_spans(
        predicted_spans=assisted_merged_spans,
        gold_spans=all_test_gold_spans,
        iou_threshold=0.50,
        label_names=LABEL_NAMES
    )
    assisted_metrics["execution_time_seconds"] = round(assist_time, 1)
    assisted_metrics["throughput_chunks_per_sec"] = round(assist_throughput, 1)

    # 5. Export Metrics and Report
    print("\n[5/5] Generating evaluation reports and metrics...")
    comparison = {
        "test_chunks": len(test_dataset),
        "total_gold_spans": len(all_test_gold_spans),
        "independent_extraction": indep_metrics,
        "classifier_assisted_extraction": assisted_metrics,
        "default_recommended_strategy": "independent" if indep_metrics["partial_match_iou50"]["macro_f1"] >= assisted_metrics["partial_match_iou50"]["macro_f1"] else "assisted"
    }

    metrics_json_path = EVALUATION_DIR / "stage4_extraction_metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)
    print(f"Test metrics saved to: {metrics_json_path}")

    # Generate Markdown Report
    generate_stage4_report(comparison)
    return comparison

def generate_stage4_report(data: Dict[str, Any]) -> None:
    """Generates evaluation/stage4_extraction_report.md."""
    indep = data["independent_extraction"]
    asst = data["classifier_assisted_extraction"]

    md = f"""# ClauseGuard Stage 4: Token-Level Span Extraction & Inference Report

## Executive Summary
This report presents the empirical evaluation of **ClauseGuard Stage 4: Token-Level Span Extraction & Inference Pipeline**.

The multi-channel RoBERTa-based span extractor was trained using the 340-contract training split and evaluated on the untouched official held-out test set (3,721 chunks across 102 contracts).

Two deployment strategies were benchmarked:
1. **Independent Extraction:** Full 9-channel span extraction across all chunks.
2. **Classifier-Assisted Extraction:** High-recall Stage 3 gating (assistance threshold $\\tau = 0.20$).

---

## 1. Strategy Comparison: Independent vs Classifier-Assisted

| Metric | Independent Extraction | Classifier-Assisted Extraction | Delta (Assisted − Indep) |
|---|---|---|---|
| **Exact-Match Macro F1** | **{indep['exact_match']['macro_f1']:.4f}** | **{asst['exact_match']['macro_f1']:.4f}** | **{asst['exact_match']['macro_f1'] - indep['exact_match']['macro_f1']:+.4f}** |
| Exact-Match Macro Precision | {indep['exact_match']['macro_precision']:.4f} | {asst['exact_match']['macro_precision']:.4f} | {asst['exact_match']['macro_precision'] - indep['exact_match']['macro_precision']:+.4f} |
| Exact-Match Macro Recall | {indep['exact_match']['macro_recall']:.4f} | {asst['exact_match']['macro_recall']:.4f} | {asst['exact_match']['macro_recall'] - indep['exact_match']['macro_recall']:+.4f} |
| **Partial-Match (IoU $\\ge$ 0.50) Macro F1** | **{indep['partial_match_iou50']['macro_f1']:.4f}** | **{asst['partial_match_iou50']['macro_f1']:.4f}** | **{asst['partial_match_iou50']['macro_f1'] - indep['partial_match_iou50']['macro_f1']:+.4f}** |
| Partial-Match Macro Precision | {indep['partial_match_iou50']['macro_precision']:.4f} | {asst['partial_match_iou50']['macro_precision']:.4f} | {asst['partial_match_iou50']['macro_precision'] - indep['partial_match_iou50']['macro_precision']:+.4f} |
| Partial-Match Macro Recall | {indep['partial_match_iou50']['macro_recall']:.4f} | {asst['partial_match_iou50']['macro_recall']:.4f} | {asst['partial_match_iou50']['macro_recall'] - indep['partial_match_iou50']['macro_recall']:+.4f} |
| Total Predicted Clauses | {indep['total_predicted_spans']:,} | {asst['total_predicted_spans']:,} | {asst['total_predicted_spans'] - indep['total_predicted_spans']:,} |
| Processing Throughput (CPU) | {indep.get('throughput_chunks_per_sec', 0.0):.1f} chunks/sec | {asst.get('throughput_chunks_per_sec', 0.0):.1f} chunks/sec | {asst.get('throughput_chunks_per_sec', 0.0) - indep.get('throughput_chunks_per_sec', 0.0):+.1f} |
| Total Execution Time | {indep.get('execution_time_seconds', 0.0):.1f} s | {asst.get('execution_time_seconds', 0.0):.1f} s | {asst.get('execution_time_seconds', 0.0) - indep.get('execution_time_seconds', 0.0):+.1f} s |

---

## 2. Category Breakdown on Official Held-Out Test Set (Independent Strategy)

| Category | Gold Count | Pred Count | Exact F1 | Exact Prec | Exact Rec | IoU $\\ge$ 0.50 F1 | IoU $\\ge$ 0.50 Prec | IoU $\\ge$ 0.50 Rec |
|---|---|---|---|---|---|---|---|---|
"""
    for cat in LABEL_NAMES:
        c = indep["per_category"][cat]
        em = c["exact_match"]
        pm = c["partial_match_iou50"]
        md += f"| `{cat}` | {c['gold_count']} | {c['pred_count']} | {em['f1']:.4f} | {em['precision']:.4f} | {em['recall']:.4f} | **{pm['f1']:.4f}** | {pm['precision']:.4f} | {pm['recall']:.4f} |\n"

    md += f"""| **MACRO AVERAGE** | **{indep['total_gold_spans']}** | **{indep['total_predicted_spans']}** | **{indep['exact_match']['macro_f1']:.4f}** | **{indep['exact_match']['macro_precision']:.4f}** | **{indep['exact_match']['macro_recall']:.4f}** | **{indep['partial_match_iou50']['macro_f1']:.4f}** | **{indep['partial_match_iou50']['macro_precision']:.4f}** | **{indep['partial_match_iou50']['macro_recall']:.4f}** |
| **MICRO AVERAGE** | **{indep['total_gold_spans']}** | **{indep['total_predicted_spans']}** | **{indep['exact_match']['micro_f1']:.4f}** | **{indep['exact_match']['micro_precision']:.4f}** | **{indep['exact_match']['micro_recall']:.4f}** | **{indep['partial_match_iou50']['micro_f1']:.4f}** | **{indep['partial_match_iou50']['micro_precision']:.4f}** | **{indep['partial_match_iou50']['micro_recall']:.4f}** |

---

## 3. Engineering Analysis & Production Recommendations

1. **Exact Match vs Partial Overlap:**
   Exact-match evaluation in legal NLP is exceptionally stringent because annotations often include or exclude surrounding introductory preambles (e.g. "Subject to the foregoing..."). Partial-match with IoU >= 0.50 accurately reflects semantic localization while respecting human annotation boundary variance.

2. **Gating Strategy Recommendation:**
   - **Recommended Default for MVP:** `{data['default_recommended_strategy'].upper()}`.
   - Independent extraction guarantees that minority clauses with subtle phrasing are never suppressed by chunk-level classification uncertainty.
   - Classifier-assisted extraction reduces CPU cycles by skipping chunks with zero candidate clause probability, serving as an optimal throughput optimization when batching large volumes.

3. **Stage 5 Handover Contract:**
   All predicted spans are structured with exact document offsets (`start_char`, `end_char`), category tags, and context windows, ready for immediate ingestion by the Stage 5 Risk Scoring Engine.
"""

    report_path = EVALUATION_DIR / "stage4_extraction_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Report written to: {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ClauseGuard Stage 4 Test Evaluation")
    parser.add_argument("--device", type=str, default="cpu", help="Device for evaluation")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    args = parser.parse_args()

    run_test_evaluation(device=args.device, batch_size=args.batch_size)
