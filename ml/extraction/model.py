"""
ClauseGuard Multi-Channel Token Span Extractor
==============================================
Transformer-based sequence tagger for legal clause extraction:
- Reuses Stage 3 fine-tuned RoBERTa encoder backbone.
- 9 independent category BIO classification channels (O=0, B=1, I=2).
- Masked cross-entropy loss with class imbalance weighting.
- Handles overlapping spans across distinct legal categories.
- Checkpoint persistence and loading interfaces.
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
from ml.extraction.token_tagging import NUM_TAGS, TAG_O, TAG_B, TAG_I, IGNORE_INDEX

class ClauseGuardSpanExtractor(nn.Module):
    """
    Token-level sequence classification architecture with 9 independent BIO channels.
    """

    def __init__(
        self,
        model_name_or_path: str = "roberta-base",
        num_categories: int = NUM_LABELS,
        num_tags: int = NUM_TAGS,
        dropout_prob: float = 0.2,
        attn_implementation: str = "eager"
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.num_categories = num_categories
        self.num_tags = num_tags
        self.dropout_prob = dropout_prob

        # Determine base architecture
        model_path = Path(model_name_or_path)
        encoder_weights_to_load = None

        if model_path.is_dir():
            cg_cfg = model_path / "clauseguard_config.json"
            sp_cfg = model_path / "span_extractor_config.json"
            if cg_cfg.exists():
                with open(cg_cfg, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                base_arch = meta.get("model_name_or_path", "roberta-base")
            elif sp_cfg.exists():
                with open(sp_cfg, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                base_arch = meta.get("base_model", "roberta-base")
            else:
                base_arch = "roberta-base"

            weights_file = model_path / "pytorch_model.bin"
            if weights_file.exists():
                encoder_weights_to_load = weights_file
        else:
            base_arch = model_name_or_path

        # Load transformer configuration with eager attention for stable CPU execution
        self.config = AutoConfig.from_pretrained(base_arch)
        self.config._attn_implementation = attn_implementation

        # Backbone transformer encoder
        self.encoder = AutoModel.from_pretrained(
            base_arch,
            config=self.config
        )

        # If checkpoint provided, transfer fine-tuned encoder weights
        if encoder_weights_to_load is not None:
            sd = torch.load(encoder_weights_to_load, map_location="cpu")
            encoder_sd = {}
            for k, v in sd.items():
                if k.startswith("encoder."):
                    encoder_sd[k[len("encoder."):]] = v
                elif not k.startswith("classifier."):
                    encoder_sd[k] = v

            if encoder_sd:
                missing, unexpected = self.encoder.load_state_dict(encoder_sd, strict=False)
                print(f"[Span Extractor] Successfully transferred {len(encoder_sd)} encoder weight tensors from {encoder_weights_to_load}")

        hidden_size = getattr(self.config, "hidden_size", 768)

        # Multi-channel token classifier head (outputs 9 * 3 logits per token)
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_categories * num_tags)

        # Initialize classifier weights
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
        Forward pass producing 9-channel BIO logits.

        Args:
            input_ids: Tensor [batch_size, seq_len]
            attention_mask: Tensor [batch_size, seq_len]
            token_type_ids: Optional Tensor for BERT architectures

        Returns:
            logits: Tensor [batch_size, seq_len, num_categories, num_tags]
        """
        encoder_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
        if token_type_ids is not None and getattr(self.encoder.config, "model_type", "") == "bert":
            encoder_kwargs["token_type_ids"] = token_type_ids

        outputs = self.encoder(**encoder_kwargs)
        sequence_output = outputs.last_hidden_state  # [B, L, H]

        dropped = self.dropout(sequence_output)
        raw_logits = self.classifier(dropped)  # [B, L, 9 * 3]

        batch_size, seq_len, _ = raw_logits.shape
        logits = raw_logits.view(batch_size, seq_len, self.num_categories, self.num_tags)
        return logits

    @staticmethod
    def predict_tags_and_probs(
        logits: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes softmax probabilities and greedy argmax tags for each channel.

        Returns:
            probs: Tensor [B, L, 9, 3]
            tags: Tensor [B, L, 9] with values in {0, 1, 2}
        """
        probs = F.softmax(logits, dim=-1)
        tags = torch.argmax(probs, dim=-1)
        return probs, tags

    @staticmethod
    def compute_loss(
        logits: torch.Tensor,
        labels: torch.Tensor,
        class_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Computes masked cross-entropy loss across all supervised categories.

        Args:
            logits: Tensor [batch_size, seq_len, num_categories, num_tags]
            labels: Tensor [batch_size, seq_len, num_categories] with values in {-100, 0, 1, 2}
            class_weights: Optional Tensor [3] for [O, B, I] class weighting.

        Returns:
            Scalar loss tensor.
        """
        batch_size, seq_len, num_cats, num_tags = logits.shape
        device = logits.device

        if class_weights is not None:
            class_weights = class_weights.to(device)

        total_loss = 0.0
        supervised_channels = 0

        for k in range(num_cats):
            cat_labels = labels[:, :, k].reshape(-1)  # [B * L]
            cat_logits = logits[:, :, k, :].reshape(-1, num_tags)  # [B * L, 3]

            valid_mask = (cat_labels != IGNORE_INDEX)
            valid_count = valid_mask.sum().item()

            if valid_count > 0:
                ce_loss = F.cross_entropy(
                    cat_logits,
                    cat_labels,
                    weight=class_weights,
                    ignore_index=IGNORE_INDEX,
                    reduction="sum"
                )
                total_loss += ce_loss / valid_count
                supervised_channels += 1

        if supervised_channels > 0:
            return total_loss / supervised_channels
        else:
            return logits.sum() * 0.0

    def save_pretrained(
        self,
        save_directory: Union[str, Path],
        tokenizer: Optional[Any] = None,
        training_metrics: Optional[Dict[str, Any]] = None
    ):
        """Serializes weights, configuration, and metadata."""
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save model weights
        torch.save(self.state_dict(), save_dir / "pytorch_model.bin")

        # Save metadata
        metadata = {
            "model_type": "roberta_span_extractor",
            "base_model": self.model_name_or_path,
            "num_categories": self.num_categories,
            "num_tags": self.num_tags,
            "label_names": LABEL_NAMES,
            "tag_names": ["O", "B", "I"],
            "training_metrics": training_metrics or {}
        }
        with open(save_dir / "span_extractor_config.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        if tokenizer is not None:
            tokenizer.save_pretrained(str(save_dir))

    @classmethod
    def from_pretrained(
        cls,
        load_directory: Union[str, Path],
        device: str = "cpu"
    ) -> Tuple["ClauseGuardSpanExtractor", Dict[str, Any]]:
        """Deserializes extractor model from checkpoint directory."""
        load_dir = Path(load_directory)
        config_file = load_dir / "span_extractor_config.json"
        weights_file = load_dir / "pytorch_model.bin"

        if not config_file.exists():
            raise FileNotFoundError(f"Missing span_extractor_config.json in {load_dir}")
        if not weights_file.exists():
            raise FileNotFoundError(f"Missing pytorch_model.bin in {load_dir}")

        with open(config_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        base_model = metadata.get("base_model", "roberta-base")
        model = cls(
            model_name_or_path=base_model,
            num_categories=metadata.get("num_categories", NUM_LABELS),
            num_tags=metadata.get("num_tags", NUM_TAGS)
        )

        state_dict = torch.load(weights_file, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        return model, metadata
