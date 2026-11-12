import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import psutil
import torch
import gc
import pandas as pd
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader

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

def print_mem(step_name):
    mem = psutil.virtual_memory()
    proc = psutil.Process()
    print(f"[{step_name}] RSS: {proc.memory_info().rss / 1e6:.1f} MB | System Avail: {mem.available / 1e9:.2f} GB")

print_mem("Start")

cfg = get_model_config("roberta", is_feasibility_run=True)
print_mem("Config loaded")

tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path)
print_mem("Tokenizer loaded")

df = pd.read_parquet(cfg.train_data_path).head(64)
val_df = pd.read_parquet(cfg.val_data_path).head(64)
print_mem("DataFrames loaded")

train_dataset = ClauseGuardDataset(df, tokenizer=tokenizer, max_length=cfg.max_length)
val_dataset = ClauseGuardDataset(val_df, tokenizer=tokenizer, max_length=cfg.max_length)
print_mem("Datasets created")

train_loader = DataLoader(
    train_dataset,
    batch_size=cfg.train_batch_size,
    shuffle=True,
    collate_fn=clauseguard_collate_fn
)
val_loader = DataLoader(
    val_dataset,
    batch_size=cfg.eval_batch_size,
    shuffle=False,
    collate_fn=clauseguard_collate_fn
)
print_mem("DataLoaders created")

pos_weight = compute_training_class_weights(cfg.train_data_path, mode="sqrt")
print_mem("Class weights computed")

model = ClauseGuardClassifier(
    model_name_or_path=cfg.model_name_or_path,
    num_labels=cfg.num_labels,
    dropout_prob=0.2,
    attn_implementation=cfg.attn_implementation
)
print_mem("Model created")

optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
print_mem("Optimizer created")

batch = next(iter(train_loader))
print_mem("Batch fetched")
print("Batch input_ids shape:", batch["input_ids"].shape)

print("Starting forward pass...")
try:
    logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    print_mem("Forward pass success")
    print("Logits shape:", logits.shape)

    loss = model.compute_loss(logits, batch["labels"], batch["label_mask"], pos_weight=pos_weight)
    print_mem("Loss computed")
    print("Loss:", loss.item())

    loss.backward()
    print_mem("Backward pass success")

    optimizer.step()
    optimizer.zero_grad()
    print_mem("Optimizer step success")

    print("\nALL TRAINING STEPS SUCCEEDED PERFECTLY!")
except Exception as e:
    import traceback
    print("Training step failed!")
    traceback.print_exc()
