import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gc
import psutil
import torch
import pandas as pd
from transformers import AutoTokenizer
from ml.training.model import ClauseGuardClassifier

# Load text using pandas
print("Loading 64 samples...")
df = pd.read_parquet("ml/data/processed/train_chunks.parquet", columns=["text", "labels", "label_mask"]).head(64)
texts = df["text"].tolist()
labels = torch.tensor(df["labels"].tolist(), dtype=torch.float32)
label_mask = torch.tensor(df["label_mask"].tolist(), dtype=torch.float32)
del df
gc.collect()

tokenizer = AutoTokenizer.from_pretrained("roberta-base")
enc = tokenizer(texts, max_length=256, padding=True, truncation=True, return_tensors="pt")
del texts, tokenizer
gc.collect()

print("Pretokenized input_ids shape:", enc["input_ids"].shape)
print("RSS before model:", psutil.Process().memory_info().rss / 1e6, "MB")

model = ClauseGuardClassifier("roberta-base", num_labels=9, attn_implementation="eager")
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, foreach=False)
pos_weight = torch.ones(9)

print("RSS after model:", psutil.Process().memory_info().rss / 1e6, "MB")

# Run 4 steps with batch_size=4
for step in range(4):
    idx = slice(step * 4, (step + 1) * 4)
    b_ids = enc["input_ids"][idx]
    b_mask = enc["attention_mask"][idx]
    b_lbl = labels[idx]
    b_lmask = label_mask[idx]

    logits = model(b_ids, b_mask)
    loss = model.compute_loss(logits, b_lbl, b_lmask, pos_weight)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(f"Step {step+1} done! Loss: {loss.item():.4f}")

print("ALL 4 STEPS COMPLETED FLAWLESSLY!")
