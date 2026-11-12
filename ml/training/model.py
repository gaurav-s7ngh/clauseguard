"""
ClauseGuard Multi-Label Transformer Classifier
==============================================
Reusable multi-label sequence classification architecture:
- Supports RoBERTa-base and InLegalBERT backbones.
- 9 output logits with sigmoid probability mapping.
- Masked BCEWithLogitsLoss excluding unverified/unknown labels.
- Numerically safe empty-mask handling.
- Checkpoint persistence and reconstruction.
"""

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig, AutoTokenizer

from ml.preprocessing.constants import NUM_LABELS, LABEL_NAMES

class ClauseGuardClassifier(nn.Module):
    """
    Multi-label sequence classifier for ClauseGuard legal document analysis.
    Outputs 9 independent logits corresponding to target legal clause categories.
    """

    def __init__(
        self,
        model_name_or_path: str = "roberta-base",
        num_labels: int = NUM_LABELS,
        dropout_prob: float = 0.2,
        attn_implementation: str = "eager"
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.num_labels = num_labels
        self.dropout_prob = dropout_prob

        # Load transformer configuration with eager attention for robust CPU execution
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.config._attn_implementation = attn_implementation
        self.config.num_labels = num_labels

        # Backbone encoder
        self.encoder = AutoModel.from_pretrained(
            model_name_or_path,
            config=self.config
        )

        hidden_size = getattr(self.config, "hidden_size", 768)

        # Classification head
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_labels)

        # Initialize head weights
        self._init_classifier_weights()

    def _init_classifier_weights(self):
        nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.constant_(self.classifier.bias, 0.0)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass producing 9 unnormalized logits.

        Args:
            input_ids: Tensor of shape [batch_size, seq_len]
            attention_mask: Tensor of shape [batch_size, seq_len]
            token_type_ids: Optional Tensor for BERT models

        Returns:
            logits: Tensor of shape [batch_size, num_labels]
        """
        encoder_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
        # BERT-based models use token_type_ids; RoBERTa does not
        if token_type_ids is not None and getattr(self.encoder.config, "model_type", "") == "bert":
            encoder_kwargs["token_type_ids"] = token_type_ids

        outputs = self.encoder(**encoder_kwargs)

        # Pooler / CLS representation (index 0 for both RoBERTa <s> and BERT [CLS])
        cls_repr = outputs.last_hidden_state[:, 0, :]

        dropped = self.dropout(cls_repr)
        logits = self.classifier(dropped)
        return logits

    @staticmethod
    def predict_proba(logits: torch.Tensor) -> torch.Tensor:
        """Applies sigmoid to logits to produce independent category probabilities."""
        return torch.sigmoid(logits)

    @staticmethod
    def compute_loss(
        logits: torch.Tensor,
        labels: torch.Tensor,
        label_mask: torch.Tensor,
        pos_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Computes masked Binary Cross Entropy with Logits loss:
        loss = sum(BCE(logits, labels) * label_mask) / max(sum(label_mask), 1.0)

        Ensures unknown/unverified labels contribute neither to the numerator
        nor the effective denominator.

        Args:
            logits: Shape [batch_size, num_labels]
            labels: Shape [batch_size, num_labels] (values in {0, 1})
            label_mask: Shape [batch_size, num_labels] (values in {0, 1})
            pos_weight: Optional shape [num_labels] tensor for class imbalance

        Returns:
            Scalar loss tensor
        """
        if pos_weight is not None:
            pos_weight = pos_weight.to(logits.device)

        # Elementwise unreduced BCE loss
        bce = F.binary_cross_entropy_with_logits(
            logits,
            labels,
            pos_weight=pos_weight,
            reduction="none"
        )

        # Apply supervision mask
        masked_bce = bce * label_mask

        # Numerically safe normalization
        mask_sum = label_mask.sum()
        if mask_sum > 0:
            loss = masked_bce.sum() / mask_sum
        else:
            # Degenerate edge case where entire batch is masked
            loss = masked_bce.sum() * 0.0

        return loss

    def save_pretrained(
        self,
        save_directory: Union[str, Path],
        tokenizer: Optional[Any] = None,
        thresholds: Optional[Dict[str, float]] = None,
        training_metrics: Optional[Dict[str, Any]] = None
    ):
        """
        Saves model weights, configuration, tokenizer, and calibration metadata.
        """
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save model weights
        weights_path = save_dir / "pytorch_model.bin"
        torch.save(self.state_dict(), weights_path)

        # Save config
        config_path = save_dir / "clauseguard_config.json"
        cfg_dict = {
            "model_name_or_path": self.model_name_or_path,
            "num_labels": self.num_labels,
            "dropout_prob": self.dropout_prob,
            "label_names": LABEL_NAMES,
            "thresholds": thresholds or {lbl: 0.5 for lbl in LABEL_NAMES},
            "training_metrics": training_metrics or {}
        }
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(cfg_dict, f, indent=2)

        # Save tokenizer if provided
        if tokenizer is not None:
            tokenizer.save_pretrained(save_dir)

    @classmethod
    def from_pretrained(
        cls,
        load_directory: Union[str, Path],
        device: str = "cpu"
    ) -> Tuple["ClauseGuardClassifier", Dict[str, Any]]:
        """
        Loads a fine-tuned ClauseGuardClassifier from checkpoint directory.
        """
        load_dir = Path(load_directory)
        config_path = load_dir / "clauseguard_config.json"
        weights_path = load_dir / "pytorch_model.bin"

        if not config_path.exists() or not weights_path.exists():
            raise FileNotFoundError(f"Missing model files in {load_dir}")

        with open(config_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        model = cls(
            model_name_or_path=meta.get("model_name_or_path", "roberta-base"),
            num_labels=meta.get("num_labels", NUM_LABELS),
            dropout_prob=meta.get("dropout_prob", 0.2)
        )

        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        return model, meta
