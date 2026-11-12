"""
ClauseGuard Stage 4 Token-Level Span Extraction Training Pipeline
=================================================================
Reproducible training script for the multi-channel legal clause span extractor:
- Reuses Stage 3 fine-tuned RoBERTa encoder backbone.
- Trains 9-channel sequence tagger with masked cross-entropy loss and class weighting.
- Evaluates on the held-out validation split with exact and partial IoU metrics.
- Serializes best model checkpoint and logs validation metrics.

Usage:
    python ml/extraction/train.py --epochs 1 --batch_size 4 --max_samples 800
    python ml/extraction/train.py --feasibility
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

# Add workspace root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ml.preprocessing.constants import LABEL_NAMES, NUM_LABELS, RANDOM_SEED
from ml.extraction.token_tagging import (
    align_spans_to_bio_labels,
    decode_spans_from_bio_tags,
    merge_overlapping_clauses,
    NUM_TAGS,
    TAG_NAMES
)
from ml.extraction.dataset import build_span_dataset, prepare_span_training_df
from ml.extraction.model import ClauseGuardSpanExtractor
from ml.extraction.evaluate import evaluate_predicted_spans, evaluate_extractor_on_dataloader

DEFAULT_BASE_MODEL = "ml/checkpoints/roberta/best_checkpoint"
CHECKPOINTS_DIR = Path("ml/checkpoints/span_extractor")
EVALUATION_DIR = Path("evaluation")

def set_seed(seed: int = RANDOM_SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_span_extractor(
    base_model_path: str = DEFAULT_BASE_MODEL,
    train_parquet: str = "ml/data/processed/train_chunks.parquet",
    val_parquet: str = "ml/data/processed/val_chunks.parquet",
    output_dir: Path = CHECKPOINTS_DIR / "best_checkpoint",
    num_epochs: int = 5,
    train_batch_size: int = 16,
    eval_batch_size: int = 32,
    gradient_accumulation_steps: int = 1,
    learning_rate: float = 4e-3,
    max_length: int = 256,
    max_train_samples: Optional[int] = 1200,
    freeze_encoder: bool = True,
    is_feasibility_run: bool = False,
    device: str = "cpu",
    seed: int = RANDOM_SEED
) -> Dict[str, Any]:
    """Executes the complete training and validation cycle for the span extractor."""
    set_seed(seed)
    print("=" * 80)
    print("CLAUSEGUARD STAGE 4: TOKEN-LEVEL SPAN EXTRACTOR TRAINING")
    print(f"Base Encoder: {base_model_path} | Device: {device.upper()} | Feasibility: {is_feasibility_run}")
    print(f"Freeze Encoder: {freeze_encoder} | Epochs: {num_epochs} | LR: {learning_rate} | Batch Size: {train_batch_size}")
    print("=" * 80)

    output_dir.mkdir(parents=True, exist_ok=True)
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Tokenizer
    print("\n[1/5] Loading tokenizer from base checkpoint...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    # 2. Prepare Training Dataset
    print(f"\n[2/5] Preparing training dataset (target max {max_train_samples} chunks)...")
    train_df = prepare_span_training_df(
        parquet_path=train_parquet,
        max_samples=32 if is_feasibility_run else max_train_samples,
        pos_ratio=0.85,
        seed=seed
    )
    print(f"Training chunks: {len(train_df):,} ({train_df['is_positive'].sum():,} positive, {(~train_df['is_positive']).sum():,} negative)")

    train_dataset = build_span_dataset(train_df, tokenizer, max_length=max_length, include_metadata=False)
    del train_df
    import gc; gc.collect()

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)

    # 3. Initialize Span Extractor Architecture
    print(f"\n[3/5] Initializing ClauseGuardSpanExtractor from {base_model_path}...")
    model = ClauseGuardSpanExtractor(
        model_name_or_path=base_model_path,
        num_categories=NUM_LABELS,
        num_tags=NUM_TAGS,
        dropout_prob=0.2,
        attn_implementation="eager"
    ).to(device)

    # Class weighting: O=0.2, B=10.0, I=3.0 (rebalanced for token class rarity)
    class_weights = torch.tensor([0.2, 10.0, 3.0], dtype=torch.float32).to(device)

    # Freeze encoder parameters for CPU training efficiency and feature preservation
    if freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad = False
        model.encoder.eval()
        trainable_params = sum(p.numel() for p in model.classifier.parameters() if p.requires_grad)
        print(f"Backbone encoder frozen. Trainable head parameters: {trainable_params:,}")
        optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=learning_rate, weight_decay=1e-4)
    else:
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate, foreach=False)

    total_training_steps = (len(train_loader) // gradient_accumulation_steps) * num_epochs
    warmup_steps = int(total_training_steps * 0.10)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max(total_training_steps, 1)
    )

    print(f"Total training steps: {total_training_steps} | Warmup steps: {warmup_steps}")

    # 4. Training Loop
    print("\n[4/5] Starting span extraction fine-tuning...")
    start_time = time.time()
    model.train()
    if freeze_encoder:
        model.encoder.eval()

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        total_epoch_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader, 1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            if freeze_encoder:
                with torch.no_grad():
                    outputs = model.encoder(input_ids=input_ids, attention_mask=attention_mask)
                    hidden_states = outputs.last_hidden_state
                raw_logits = model.classifier(model.dropout(hidden_states))
                batch_size_i, seq_len_i, _ = raw_logits.shape
                logits = raw_logits.view(batch_size_i, seq_len_i, NUM_LABELS, NUM_TAGS)
            else:
                logits = model(input_ids, attention_mask)

            loss = model.compute_loss(logits, labels, class_weights=class_weights)
            loss = loss / gradient_accumulation_steps
            loss.backward()
            total_epoch_loss += loss.item() * gradient_accumulation_steps

            del logits, loss

            if step % gradient_accumulation_steps == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if step % 25 == 0 or step == len(train_loader):
                curr_loss = total_epoch_loss / step
                print(f"  Epoch {epoch}/{num_epochs} | Step {step}/{len(train_loader)} | Token Loss: {curr_loss:.4f}", flush=True)

        epoch_time = time.time() - epoch_start
        print(f"  Epoch {epoch} finished in {epoch_time:.1f}s | Average Loss: {total_epoch_loss / len(train_loader):.4f}")

    total_train_time = time.time() - start_time
    print(f"\nTraining completed in {total_train_time / 60:.2f} minutes.")

    # Save model checkpoint
    print(f"\nSaving best span extractor checkpoint to {output_dir}...")
    model.save_pretrained(
        save_directory=output_dir,
        tokenizer=tokenizer,
        training_metrics={"train_loss": total_epoch_loss / len(train_loader), "train_time_sec": round(total_train_time, 1)}
    )

    # 5. Validation Evaluation
    print("\n[5/5] Running validation evaluation on held-out validation set...")
    val_cols = ["chunk_id", "contract_id", "text", "labels", "label_mask", "char_start", "char_end", "is_positive", "spans", "span_categories"]
    val_df = pd.read_parquet(val_parquet, columns=val_cols)
    if is_feasibility_run:
        val_df = val_df.head(32)

    # Extract all gold spans in validation split (deduplicated per contract)
    all_gold_spans = []
    seen_gold = set()
    for _, row in val_df.iterrows():
        c_id = row["contract_id"]
        for s in row["spans"]:
            key = (c_id, s["category"], s["global_start"], s["global_end"])
            if key not in seen_gold:
                seen_gold.add(key)
                g = dict(s)
                g["contract_id"] = c_id
                g["document_id"] = c_id
                all_gold_spans.append(g)

    print(f"Loaded {len(val_df):,} validation chunks with {len(all_gold_spans):,} unique gold clause spans.")
    val_dataset = build_span_dataset(val_df, tokenizer, max_length=max_length, include_metadata=True)
    del val_df
    gc.collect()

    val_loader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False)

    pred_spans_raw, val_loss = evaluate_extractor_on_dataloader(
        model=model,
        dataloader=val_loader,
        device=device,
        min_confidence=0.20,
        b_threshold=0.15,
        span_threshold=0.20,
        class_weights=class_weights,
        verbose=True
    )

    print(f"Raw extracted spans: {len(pred_spans_raw):,}. Merging overlapping/cross-chunk predictions...")
    pred_spans_merged = merge_overlapping_clauses(pred_spans_raw)
    print(f"Deduplicated merged spans: {len(pred_spans_merged):,}")

    eval_results = evaluate_predicted_spans(
        predicted_spans=pred_spans_merged,
        gold_spans=all_gold_spans,
        iou_threshold=0.50,
        label_names=LABEL_NAMES
    )
    eval_results["validation_loss"] = round(val_loss, 4)
    eval_results["training_time_seconds"] = round(total_train_time, 1)
    eval_results["checkpoint_dir"] = str(output_dir)

    # Print evaluation table
    print("\n" + "=" * 80)
    print("SPAN EXTRACTION VALIDATION RESULTS (EXACT MATCH vs PARTIAL OVERLAP IoU >= 0.50)")
    print("=" * 80)
    print(f"{'Category':<28} | {'Golds':<6} | {'Preds':<6} | {'Exact F1':<9} | {'Exact P':<8} | {'Exact R':<8} | {'IoU50 F1':<9}")
    print("-" * 88)
    for cat in LABEL_NAMES:
        c = eval_results["per_category"][cat]
        em = c["exact_match"]
        pm = c["partial_match_iou50"]
        print(f"{cat:<28} | {c['gold_count']:<6} | {c['pred_count']:<6} | {em['f1']:<9.4f} | {em['precision']:<8.4f} | {em['recall']:<8.4f} | {pm['f1']:<9.4f}")
    print("-" * 88)
    em_macro = eval_results["exact_match"]
    pm_macro = eval_results["partial_match_iou50"]
    print(f"{'MACRO AVERAGE':<28} | {eval_results['total_gold_spans']:<6} | {eval_results['total_predicted_spans']:<6} | {em_macro['macro_f1']:<9.4f} | {em_macro['macro_precision']:<8.4f} | {em_macro['macro_recall']:<8.4f} | {pm_macro['macro_f1']:<9.4f}")
    print(f"{'MICRO AVERAGE':<28} | {eval_results['total_gold_spans']:<6} | {eval_results['total_predicted_spans']:<6} | {em_macro['micro_f1']:<9.4f} | {em_macro['micro_precision']:<8.4f} | {em_macro['micro_recall']:<8.4f} | {pm_macro['micro_f1']:<9.4f}")

    metrics_out = EVALUATION_DIR / "stage4_val_metrics.json"
    with open(metrics_out, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)
    print(f"\nValidation metrics saved to: {metrics_out}")

    return eval_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ClauseGuard Stage 4 Span Extraction Training")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=4e-3, help="Learning rate")
    parser.add_argument("--max_samples", type=int, default=1200, help="Max training samples")
    parser.add_argument("--no_freeze_encoder", action="store_true", help="Unfreeze encoder backbone")
    parser.add_argument("--feasibility", action="store_true", help="Quick feasibility run on 32 samples")
    args = parser.parse_args()

    train_span_extractor(
        num_epochs=args.epochs,
        train_batch_size=args.batch_size,
        learning_rate=args.lr,
        max_train_samples=args.max_samples,
        freeze_encoder=not args.no_freeze_encoder,
        is_feasibility_run=args.feasibility
    )
