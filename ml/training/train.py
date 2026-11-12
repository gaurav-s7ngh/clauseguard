"""
ClauseGuard Multi-Label Transformer Training Pipeline
======================================================
Unified, reproducible training script for ClauseGuard models:
- Trains RoBERTa-base (baseline) and InLegalBERT (challenger).
- Implements masked BCE loss with square-root positive class weighting.
- Evaluates on the full unsampled validation set (2,812 chunks).
- Performs threshold calibration on validation predictions.
- Saves best checkpoints and logs training metrics.

Usage:
    python ml/training/train.py --model roberta
    python ml/training/train.py --model inlegalbert
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
from ml.preprocessing.dataset import ClauseGuardDataset, clauseguard_collate_fn
from ml.training.config import (
    TrainingConfig,
    get_model_config,
    compute_training_class_weights,
    CHECKPOINTS_DIR,
    EVALUATION_DIR,
)
from ml.training.model import ClauseGuardClassifier
from ml.training.evaluate import (
    evaluate_model_on_dataloader,
    calibrate_thresholds,
    compute_multilabel_metrics,
)

def set_seed(seed: int = RANDOM_SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def prepare_train_dataframe(
    train_parquet_path: Path,
    max_samples: Optional[int] = None,
    seed: int = RANDOM_SEED
) -> pd.DataFrame:
    """
    Loads training chunks. If max_samples is set for CPU training efficiency:
    - Retains 60% positive chunks across all 9 categories (stratified representation).
    - Subsamples negative chunks to reach max_samples (40% negative).
    - Shuffles deterministically.
    """
    columns_to_load = ["chunk_id", "contract_id", "text", "labels", "label_mask", "char_start", "char_end", "is_positive", "chunk_index"]
    df = pd.read_parquet(train_parquet_path, columns=columns_to_load)
    if max_samples is None or len(df) <= max_samples:
        return df

    pos_df = df[df["is_positive"] == True]
    neg_df = df[df["is_positive"] == False]

    n_pos_target = int(max_samples * 0.60)
    n_neg_target = max_samples - n_pos_target

    if len(pos_df) > n_pos_target:
        sampled_pos = pos_df.sample(n=n_pos_target, random_state=seed)
    else:
        sampled_pos = pos_df

    sampled_neg = neg_df.sample(n=min(n_neg_target, len(neg_df)), random_state=seed)
    combined = pd.concat([sampled_pos, sampled_neg]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    del df, pos_df, neg_df, sampled_pos, sampled_neg
    import gc; gc.collect()
    return combined

class PretokenizedClauseGuardDataset(torch.utils.data.Dataset):
    """
    In-memory tensor dataset for maximum CPU training speed and zero heap fragmentation.
    """
    def __init__(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        label_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels
        self.label_mask = label_mask
        self.token_type_ids = token_type_ids

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
            "label_mask": self.label_mask[idx]
        }
        if self.token_type_ids is not None:
            item["token_type_ids"] = self.token_type_ids[idx]
        return item

def build_pretokenized_dataset(
    df: pd.DataFrame,
    tokenizer: AutoTokenizer,
    max_length: int
) -> PretokenizedClauseGuardDataset:
    texts = df["text"].tolist()
    enc = tokenizer(
        texts,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    labels = torch.tensor(np.array(df["labels"].tolist()), dtype=torch.float32)
    label_mask = torch.tensor(np.array(df["label_mask"].tolist()), dtype=torch.float32)

    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    token_type_ids = enc.get("token_type_ids")

    del texts, enc
    import gc; gc.collect()

    return PretokenizedClauseGuardDataset(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        label_mask=label_mask,
        token_type_ids=token_type_ids
    )

def train_model(
    config: TrainingConfig,
    is_feasibility_run: bool = False
) -> Dict[str, Any]:
    """
    Executes complete training and validation cycle for a transformer model.
    """
    set_seed(config.seed)
    print("=" * 80)
    print(f"CLAUSEGUARD TRAINING: {config.model_name_or_path.upper()} ({config.model_alias.upper()})")
    print(f"Device: {config.device.upper()} | Feasibility run: {is_feasibility_run}")
    print("=" * 80)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.eval_output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load tokenizer
    print("\n[1/6] Loading model tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)

    # 2. Prepare datasets
    print("\n[2/6] Preparing training dataset...")
    train_df = prepare_train_dataframe(
        train_parquet_path=config.train_data_path,
        max_samples=config.max_train_samples,
        seed=config.seed
    )
    if is_feasibility_run:
        train_df = train_df.head(64)

    # Get sample text for latency evaluation later
    val_cols = ["chunk_id", "contract_id", "text", "labels", "label_mask", "char_start", "char_end", "is_positive", "chunk_index"]
    sample_df = pd.read_parquet(config.val_data_path, columns=["text"])
    sample_text = str(sample_df.iloc[0]["text"])
    del sample_df
    import gc; gc.collect()

    print(f"Training dataset: {len(train_df):,} chunks ({train_df['is_positive'].sum():,} positive, {(~train_df['is_positive']).sum():,} negative)")

    train_dataset = build_pretokenized_dataset(train_df, tokenizer, max_length=config.max_length)
    del train_df
    gc.collect()

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True
    )

    # 3. Class weights
    print("\n[3/6] Computing training class weights...")
    pos_weight = compute_training_class_weights(
        config.train_data_path,
        mode=config.pos_weight_mode,
        cap=config.pos_weight_cap
    ).to(config.device)

    print("Computed positive weights (BCE pos_weight):")
    for idx, name in enumerate(LABEL_NAMES):
        print(f"  {name:<28}: {pos_weight[idx].item():.2f}")

    # 4. Initialize model
    print(f"\n[4/6] Initializing {config.model_name_or_path} classifier...")
    model = ClauseGuardClassifier(
        model_name_or_path=config.model_name_or_path,
        num_labels=config.num_labels,
        dropout_prob=0.2,
        attn_implementation=config.attn_implementation
    ).to(config.device)

    # 5. Optimizer & Scheduler
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": config.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=config.learning_rate, foreach=False)

    total_training_steps = (len(train_loader) // config.gradient_accumulation_steps) * config.num_epochs
    warmup_steps = int(total_training_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max(total_training_steps, 1)
    )

    print(f"Total training steps: {total_training_steps} | Warmup steps: {warmup_steps}")

    # 6. Training loop
    print("\n[5/6] Starting training...")
    best_val_macro_f1 = -1.0
    best_checkpoint_dir = config.output_dir / "best_checkpoint"
    training_history = []
    start_train_time = time.time()

    for epoch in range(1, config.num_epochs + 1):
        epoch_start = time.time()
        model.train()
        total_epoch_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader, 1):
            input_ids = batch["input_ids"].to(config.device)
            attention_mask = batch["attention_mask"].to(config.device)
            labels = batch["labels"].to(config.device)
            label_mask = batch["label_mask"].to(config.device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(config.device)

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )

            loss = model.compute_loss(
                logits=logits,
                labels=labels,
                label_mask=label_mask,
                pos_weight=pos_weight
            )

            # Normalize loss for gradient accumulation
            loss = loss / config.gradient_accumulation_steps
            loss.backward()
            total_epoch_loss += loss.item() * config.gradient_accumulation_steps

            if step % config.gradient_accumulation_steps == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if step % 25 == 0 or step == len(train_loader):
                curr_loss = total_epoch_loss / step
                print(f"  Epoch {epoch}/{config.num_epochs} | Step {step}/{len(train_loader)} | Loss: {curr_loss:.4f}")

        epoch_loss = total_epoch_loss / len(train_loader)
        epoch_time = time.time() - epoch_start

        # Evaluate on validation set
        print(f"  --> Loading and evaluating on validation set (Epoch {epoch})...")
        val_df = pd.read_parquet(config.val_data_path, columns=val_cols)
        if is_feasibility_run:
            val_df = val_df.head(64)
        print(f"  Validation dataset: {len(val_df):,} chunks (FULL unsampled distribution)")
        val_dataset = build_pretokenized_dataset(val_df, tokenizer, max_length=config.max_length)
        del val_df
        gc.collect()
        val_loader = DataLoader(val_dataset, batch_size=config.eval_batch_size, shuffle=False)

        eval_metrics, val_probs, val_labels, val_masks = evaluate_model_on_dataloader(
            model=model,
            dataloader=val_loader,
            device=config.device,
            thresholds=0.50,
            pos_weight=pos_weight
        )

        del val_loader, val_dataset
        gc.collect()

        val_macro_f1 = eval_metrics["macro_f1"]
        val_micro_f1 = eval_metrics["micro_f1"]
        val_loss = eval_metrics["loss"]

        print(f"  Epoch {epoch} complete in {epoch_time:.1f}s | Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f} | Val Macro F1: {val_macro_f1:.4f} | Val Micro F1: {val_micro_f1:.4f}")

        training_history.append({
            "epoch": epoch,
            "train_loss": round(epoch_loss, 4),
            "val_loss": val_loss,
            "val_macro_f1": val_macro_f1,
            "val_micro_f1": val_micro_f1,
            "epoch_time_seconds": round(epoch_time, 1)
        })

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_val_probs = val_probs
            best_val_labels = val_labels
            best_val_masks = val_masks
            best_val_metrics = eval_metrics
            print(f"  ★ New best validation Macro F1: {best_val_macro_f1:.4f}! Saving checkpoint...")
            model.save_pretrained(
                save_directory=best_checkpoint_dir,
                tokenizer=tokenizer,
                thresholds={lbl: 0.50 for lbl in LABEL_NAMES},
                training_metrics={"best_macro_f1": best_val_macro_f1, "epoch": epoch}
            )

    total_train_time = time.time() - start_train_time
    print(f"\nTraining completed in {total_train_time/60:.2f} minutes.")

    # 6. Final Evaluation & Threshold Calibration on Best Model
    print("\n[6/6] Calibrating decision thresholds on validation predictions...")
    best_model, meta = ClauseGuardClassifier.from_pretrained(
        load_directory=best_checkpoint_dir,
        device=config.device
    )

    # Use best validation outputs cached from training loop, or evaluate if absent
    if 'best_val_probs' in locals() and best_val_probs is not None:
        val_metrics_default = best_val_metrics
        val_probs = best_val_probs
        val_labels = best_val_labels
        val_masks = best_val_masks
    else:
        val_metrics_default, val_probs, val_labels, val_masks = evaluate_model_on_dataloader(
            model=best_model,
            dataloader=val_loader,
            device=config.device,
            thresholds=0.50,
            pos_weight=pos_weight
        )

    optimal_thresholds, calib_f1s = calibrate_thresholds(
        probs=val_probs,
        labels=val_labels,
        label_mask=val_masks,
        label_names=LABEL_NAMES
    )

    print("\nOptimal Calibrated Thresholds & F1 Comparison:")
    print(f"{'Category':<28} | {'Default (0.50) F1':<18} | {'Optimal Threshold':<18} | {'Calibrated F1':<15}")
    print("-" * 85)
    for cat in LABEL_NAMES:
        def_f1 = val_metrics_default["per_category"][cat]["f1"]
        opt_t = optimal_thresholds[cat]
        cal_f1 = calib_f1s[cat]
        print(f"{cat:<28} | {def_f1:<18.4f} | {opt_t:<18.2f} | {cal_f1:<15.4f}")

    # Evaluate with calibrated thresholds
    val_metrics_calibrated = compute_multilabel_metrics(
        probs=val_probs,
        labels=val_labels,
        label_mask=val_masks,
        thresholds=optimal_thresholds,
        label_names=LABEL_NAMES
    )
    val_metrics_calibrated["loss"] = val_metrics_default["loss"]

    print("\nValidation Summary:")
    print(f"  Default Macro F1:    {val_metrics_default['macro_f1']:.4f} | Micro F1: {val_metrics_default['micro_f1']:.4f}")
    print(f"  Calibrated Macro F1: {val_metrics_calibrated['macro_f1']:.4f} | Micro F1: {val_metrics_calibrated['micro_f1']:.4f}")

    # Re-save best model with calibrated thresholds
    best_model.save_pretrained(
        save_directory=best_checkpoint_dir,
        tokenizer=tokenizer,
        thresholds=optimal_thresholds,
        training_metrics={
            "best_macro_f1_default": val_metrics_default["macro_f1"],
            "best_macro_f1_calibrated": val_metrics_calibrated["macro_f1"],
            "training_time_seconds": round(total_train_time, 1),
            "history": training_history
        }
    )

    # Measure inference latency
    print("\nMeasuring inference latency (100 single-chunk forward passes)...")
    latencies = []
    best_model.eval()
    tok_sample = tokenizer(sample_text, max_length=config.max_length, padding="max_length", truncation=True, return_tensors="pt")
    in_ids = tok_sample["input_ids"].to(config.device)
    att_mask = tok_sample["attention_mask"].to(config.device)
    tok_type_ids = tok_sample.get("token_type_ids")
    if tok_type_ids is not None:
        tok_type_ids = tok_type_ids.to(config.device)

    with torch.no_grad():
        for _ in range(100):
            t0 = time.time()
            _ = best_model(in_ids, att_mask, token_type_ids=tok_type_ids)
            latencies.append((time.time() - t0) * 1000.0)

    p50_latency = float(np.median(latencies))
    p95_latency = float(np.percentile(latencies, 95))
    print(f"  Latency p50: {p50_latency:.2f} ms | p95: {p95_latency:.2f} ms per chunk on {config.device.upper()}")

    # Model file size
    weights_path = best_checkpoint_dir / "pytorch_model.bin"
    model_size_mb = round(weights_path.stat().st_size / (1024 * 1024), 2) if weights_path.exists() else 0.0

    result = {
        "model_alias": config.model_alias,
        "model_name_or_path": config.model_name_or_path,
        "device": config.device,
        "total_train_time_seconds": round(total_train_time, 1),
        "model_size_mb": model_size_mb,
        "latency_ms": {
            "p50": round(p50_latency, 2),
            "p95": round(p95_latency, 2)
        },
        "training_history": training_history,
        "validation_metrics_default_05": val_metrics_default,
        "validation_metrics_calibrated": val_metrics_calibrated,
        "calibrated_thresholds": optimal_thresholds,
        "checkpoint_dir": str(best_checkpoint_dir)
    }

    metrics_out = config.eval_output_dir / f"{config.model_alias}_val_metrics.json"
    with open(metrics_out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Validation results saved to {metrics_out}")

    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ClauseGuard Stage 3 Training")
    parser.add_argument("--model", type=str, default="roberta", choices=["roberta", "inlegalbert"], help="Model to train")
    parser.add_argument("--feasibility", action="store_true", help="Run quick feasibility smoke test")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--max_samples", type=int, default=None, help="Max training samples")
    args = parser.parse_args()

    cfg = get_model_config(args.model, is_feasibility_run=args.feasibility)
    if args.epochs is not None:
        cfg.num_epochs = args.epochs
    if args.batch_size is not None:
        cfg.train_batch_size = args.batch_size
    if args.max_samples is not None:
        cfg.max_train_samples = args.max_samples
    cfg.learning_rate = args.lr

    train_model(cfg, is_feasibility_run=args.feasibility)
