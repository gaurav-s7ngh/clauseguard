"""
ClauseGuard Stage 3 Training Configuration
===========================================
Central configuration for multi-label transformer training:
- Hardware detection (CPU vs CUDA)
- Model hyperparameters for RoBERTa-base and InLegalBERT
- Class imbalance weights calculation and capping
- Dynamic batching, learning rate, and scheduler settings
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import torch
import numpy as np
import pandas as pd

from ml.preprocessing.constants import LABEL_NAMES, NUM_LABELS, RANDOM_SEED

# Hardware detection
HAS_CUDA: bool = torch.cuda.is_available()
DEFAULT_DEVICE: str = "cuda" if HAS_CUDA else "cpu"

PROCESSED_DATA_DIR = Path("ml/data/processed")
CHECKPOINTS_DIR = Path("ml/checkpoints")
EVALUATION_DIR = Path("evaluation")

@dataclass
class TrainingConfig:
    # Model specification
    model_name_or_path: str = "roberta-base"
    model_alias: str = "roberta"  # 'roberta' or 'inlegalbert'
    num_labels: int = NUM_LABELS
    label_names: List[str] = field(default_factory=lambda: list(LABEL_NAMES))

    # Paths
    train_data_path: Path = PROCESSED_DATA_DIR / "train_chunks.parquet"
    val_data_path: Path = PROCESSED_DATA_DIR / "val_chunks.parquet"
    test_data_path: Path = PROCESSED_DATA_DIR / "test_chunks.parquet"
    output_dir: Path = CHECKPOINTS_DIR / "roberta"
    eval_output_dir: Path = EVALUATION_DIR

    # Training parameters
    device: str = DEFAULT_DEVICE
    max_length: int = 256  # Covers 90%+ of legal chunk tokens; prevents memory allocator failures on CPU
    train_batch_size: int = 16 if HAS_CUDA else 4
    eval_batch_size: int = 32
    gradient_accumulation_steps: int = 4 if not HAS_CUDA else 1
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.10
    max_grad_norm: float = 1.0
    num_epochs: int = 1

    # Training efficiency sampling (applied ONLY to training data; val and test are 100% full distribution)
    # 600 positives + 400 negatives = 1,000 chunks for balanced, fast CPU convergence (~15 min training)
    max_train_samples: Optional[int] = 1000 if not HAS_CUDA else None
    pos_neg_ratio: float = 1.0 / 1.5  # ~40% positive, 60% negative in training subsample

    # Class imbalance weighting mode: 'sqrt', 'capped', 'none'
    pos_weight_mode: str = "sqrt"
    pos_weight_cap: float = 10.0

    # Early stopping and saving
    early_stopping_patience: int = 2
    save_best_model: bool = True
    seed: int = RANDOM_SEED

    # Attention backend (eager attention is rock-solid on CPU and avoids SDPA allocation bugs)
    attn_implementation: str = "eager"

def compute_training_class_weights(
    train_parquet_path: Path = PROCESSED_DATA_DIR / "train_chunks.parquet",
    mode: str = "sqrt",
    cap: float = 10.0
) -> torch.Tensor:
    """
    Computes class imbalance weights pos_weight for BCEWithLogitsLoss exclusively from training data.
    Modes:
        'sqrt': sqrt(neg / pos) (smoothly scales between ~5.7 and ~10.2)
        'capped': min(neg / pos, cap)
        'none': vector of 1.0s
    """
    if not train_parquet_path.exists():
        return torch.ones(NUM_LABELS, dtype=torch.float32)

    df = pd.read_parquet(train_parquet_path, columns=["labels", "label_mask"])
    labels = np.array(df["labels"].tolist(), dtype=float)
    masks = np.array(df["label_mask"].tolist(), dtype=float)
    del df

    pos_counts = (labels * masks).sum(axis=0)
    neg_counts = ((1.0 - labels) * masks).sum(axis=0)

    # Avoid divide by zero
    pos_counts = np.maximum(pos_counts, 1.0)
    raw_weights = neg_counts / pos_counts

    if mode == "sqrt":
        weights = np.sqrt(raw_weights)
    elif mode == "capped":
        weights = np.clip(raw_weights, 1.0, cap)
    elif mode == "none":
        weights = np.ones_like(raw_weights)
    else:
        raise ValueError(f"Unknown pos_weight_mode: {mode}")

    # Clamp to reasonable bounds to prevent exploding gradients
    weights = np.clip(weights, 1.0, cap)
    return torch.tensor(weights, dtype=torch.float32)

def get_model_config(model_alias: str, is_feasibility_run: bool = False) -> TrainingConfig:
    """
    Factory function returning model-specific configuration.
    """
    if model_alias.lower() in ["roberta", "roberta-base"]:
        cfg = TrainingConfig(
            model_name_or_path="roberta-base",
            model_alias="roberta",
            output_dir=CHECKPOINTS_DIR / "roberta",
            learning_rate=3e-5
        )
    elif model_alias.lower() in ["inlegalbert", "legalbert"]:
        cfg = TrainingConfig(
            model_name_or_path="law-ai/InLegalBERT",
            model_alias="inlegalbert",
            output_dir=CHECKPOINTS_DIR / "inlegalbert",
            learning_rate=3e-5
        )
    else:
        raise ValueError(f"Unsupported model alias: {model_alias}")

    if is_feasibility_run:
        cfg.max_train_samples = 64
        cfg.num_epochs = 1
        cfg.eval_batch_size = 16

    return cfg
