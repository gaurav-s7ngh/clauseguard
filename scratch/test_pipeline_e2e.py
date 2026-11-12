import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gc
import psutil
import torch
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from ml.preprocessing.constants import LABEL_NAMES, NUM_LABELS, RANDOM_SEED
from ml.training.model import ClauseGuardClassifier
from ml.training.evaluate import calibrate_thresholds, compute_multilabel_metrics

print("Testing end-to-end tensor pipeline...")

tokenizer = AutoTokenizer.from_pretrained("roberta-base")

# Load 32 train, 32 val
df_tr = pd.read_parquet("ml/data/processed/train_chunks.parquet", columns=["text", "labels", "label_mask"]).head(32)
enc_tr = tokenizer(df_tr["text"].tolist(), max_length=256, padding="max_length", truncation=True, return_tensors="pt")
lbl_tr = torch.tensor(np.array(df_tr["labels"].tolist()), dtype=torch.float32)
mask_tr = torch.tensor(np.array(df_tr["label_mask"].tolist()), dtype=torch.float32)
del df_tr; gc.collect()

train_ds = TensorDataset(enc_tr["input_ids"], enc_tr["attention_mask"], lbl_tr, mask_tr)
train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)

df_val = pd.read_parquet("ml/data/processed/val_chunks.parquet", columns=["text", "labels", "label_mask"]).head(32)
enc_val = tokenizer(df_val["text"].tolist(), max_length=256, padding="max_length", truncation=True, return_tensors="pt")
lbl_val = torch.tensor(np.array(df_val["labels"].tolist()), dtype=torch.float32)
mask_val = torch.tensor(np.array(df_val["label_mask"].tolist()), dtype=torch.float32)
del df_val; gc.collect()

val_ds = TensorDataset(enc_val["input_ids"], enc_val["attention_mask"], lbl_val, mask_val)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

model = ClauseGuardClassifier("roberta-base", num_labels=9, attn_implementation="eager")
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, foreach=False)
pos_weight = torch.ones(9)

# 1 train epoch
model.train()
for b_ids, b_mask, b_lbl, b_lmask in train_loader:
    logits = model(b_ids, b_mask)
    loss = model.compute_loss(logits, b_lbl, b_lmask, pos_weight)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
print("Train loop finished successfully!")

# Validation eval
model.eval()
val_probs, val_targets, val_masks = [], [], []
with torch.no_grad():
    for b_ids, b_mask, b_lbl, b_lmask in val_loader:
        logits = model(b_ids, b_mask)
        probs = model.predict_proba(logits)
        val_probs.append(probs.numpy())
        val_targets.append(b_lbl.numpy())
        val_masks.append(b_lmask.numpy())

all_probs = np.vstack(val_probs)
all_targets = np.vstack(val_targets)
all_masks = np.vstack(val_masks)

metrics_def = compute_multilabel_metrics(all_probs, all_targets, all_masks, thresholds=0.5)
print(f"Default Macro F1: {metrics_def['macro_f1']:.4f}")

opt_thresh, calib_f1s = calibrate_thresholds(all_probs, all_targets, all_masks, LABEL_NAMES)
print("Optimal thresholds calibrated:", opt_thresh)

metrics_cal = compute_multilabel_metrics(all_probs, all_targets, all_masks, thresholds=opt_thresh)
print(f"Calibrated Macro F1: {metrics_cal['macro_f1']:.4f}")

# Test saving and loading
save_path = Path("scratch/test_ckpt")
model.save_pretrained(save_path, tokenizer=tokenizer, thresholds=opt_thresh)
loaded_model, meta = ClauseGuardClassifier.from_pretrained(save_path)
print("Model reloaded successfully!")
print("Reloaded thresholds:", meta.get("thresholds"))

print("\nPIPELINE VERIFIED END-TO-END 100% OPERATIONAL!")
