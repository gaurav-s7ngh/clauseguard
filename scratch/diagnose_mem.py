import os, psutil, gc, torch
import pandas as pd
from transformers import AutoTokenizer, AutoConfig, AutoModel
import torch.nn as nn

def print_mem(step_name):
    p = psutil.Process()
    info = p.memory_info()
    print(f"[{step_name}] RSS: {info.rss / (1024*1024):.1f} MB, VMS: {info.vms / (1024*1024):.1f} MB")

print_mem("Start")
tok = AutoTokenizer.from_pretrained("roberta-base")
print_mem("After tokenizer")

train_df = pd.read_parquet("ml/data/processed/train_chunks.parquet", columns=["text", "labels", "label_mask", "is_positive"])
pos_df = train_df[train_df["is_positive"] == True].sample(n=600, random_state=42)
neg_df = train_df[train_df["is_positive"] == False].sample(n=400, random_state=42)
train_df = pd.concat([pos_df, neg_df]).sample(frac=1.0, random_state=42).reset_index(drop=True)
del pos_df, neg_df
gc.collect()

texts = train_df["text"].tolist()
enc_train = tok(texts, max_length=256, padding="max_length", truncation=True, return_tensors="pt")
del texts, train_df
gc.collect()
print_mem("After train pretokenization (1000 rows)")

# Now add val_df (2812 rows)
val_df = pd.read_parquet("ml/data/processed/val_chunks.parquet", columns=["text", "labels", "label_mask", "is_positive"])
val_texts = val_df["text"].tolist()
enc_val = tok(val_texts, max_length=256, padding="max_length", truncation=True, return_tensors="pt")
del val_texts, val_df
gc.collect()
print_mem("After val pretokenization (2812 rows)")

# Model
cfg = AutoConfig.from_pretrained("roberta-base")
cfg._attn_implementation = "eager"
cfg.num_labels = 9
encoder = AutoModel.from_pretrained("roberta-base", config=cfg)
classifier = nn.Linear(768, 9)
print_mem("After model loaded")

optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(classifier.parameters()), lr=3e-5, foreach=False)
print_mem("After AdamW init")

# Run batch 0
b_ids = enc_train["input_ids"][:4]
b_mask = enc_train["attention_mask"][:4]

print("Running forward pass...")
out = encoder(input_ids=b_ids, attention_mask=b_mask)
print_mem("After forward pass")

cls_tok = out.last_hidden_state[:, 0, :]
logits = classifier(cls_tok)
loss = logits.sum()
print("Running backward pass...")
loss.backward()
print_mem("After backward pass")
print("SUCCESS WITH BOTH TRAIN AND VAL IN MEMORY!")
