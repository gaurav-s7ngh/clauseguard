"""
ClauseGuard Inference & Model Loading Engine
============================================
Lightweight inference pipeline for serving fine-tuned ClauseGuard transformer models:
- Loads model weights, configuration, tokenizer, and frozen calibrated thresholds
- Executes forward pass with sigmoid probability calibration
- Returns detected categories, probabilities, and decision flags
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List, Union, Optional
import torch
from transformers import AutoTokenizer

from ml.preprocessing.constants import LABEL_NAMES, NUM_LABELS
from ml.training.model import ClauseGuardClassifier

class ClauseGuardInference:
    """
    Inference interface for ClauseGuard multi-label legal clause classification.
    """

    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        device: Optional[str] = None
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        config_path = self.checkpoint_dir / "clauseguard_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Missing clauseguard_config.json in {self.checkpoint_dir}")

        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        self.label_names = self.config.get("label_names", LABEL_NAMES)
        self.thresholds = self.config.get("thresholds", {lbl: 0.50 for lbl in self.label_names})

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.checkpoint_dir))

        # Load model
        self.model, _ = ClauseGuardClassifier.from_pretrained(
            load_directory=self.checkpoint_dir,
            device=self.device
        )
        self.model.eval()

    def predict(
        self,
        text: str,
        max_length: int = 512
    ) -> Dict[str, Any]:
        """
        Runs clause classification on a single legal text chunk.

        Args:
            text: Raw legal contract text.
            max_length: Maximum sequence length.

        Returns:
            Dict containing detected categories, category probabilities, and binary flags.
        """
        encoding = self.tokenizer(
            text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        token_type_ids = encoding.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(self.device)

        with torch.no_grad():
            logits = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            probs = self.model.predict_proba(logits).squeeze(0).cpu().numpy()

        results = {}
        detected_categories = []

        for idx, cat_name in enumerate(self.label_names):
            p = float(probs[idx])
            t = float(self.thresholds.get(cat_name, 0.50))
            is_detected = (p >= t)
            if is_detected:
                detected_categories.append(cat_name)

            results[cat_name] = {
                "detected": is_detected,
                "probability": round(p, 4),
                "threshold": round(t, 4)
            }

        return {
            "detected_categories": detected_categories,
            "has_clause": len(detected_categories) > 0,
            "categories": results
        }

    def predict_batch(
        self,
        texts: List[str],
        max_length: int = 512,
        batch_size: int = 16
    ) -> List[Dict[str, Any]]:
        """
        Runs batch classification over a list of text chunks.
        """
        batch_results = []
        for i in range(0, len(texts), batch_size):
            chunk_batch = texts[i : i + batch_size]
            encoding = self.tokenizer(
                chunk_batch,
                max_length=max_length,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )

            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)
            token_type_ids = encoding.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(self.device)

            with torch.no_grad():
                logits = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
                probs = self.model.predict_proba(logits).cpu().numpy()

            for b_idx in range(len(chunk_batch)):
                res = {}
                detected = []
                for cat_idx, cat_name in enumerate(self.label_names):
                    p = float(probs[b_idx, cat_idx])
                    t = float(self.thresholds.get(cat_name, 0.50))
                    is_det = (p >= t)
                    if is_det:
                        detected.append(cat_name)
                    res[cat_name] = {
                        "detected": is_det,
                        "probability": round(p, 4),
                        "threshold": round(t, 4)
                    }
                batch_results.append({
                    "detected_categories": detected,
                    "has_clause": len(detected) > 0,
                    "categories": res
                })

        return batch_results
