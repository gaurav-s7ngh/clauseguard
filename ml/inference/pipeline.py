"""
ClauseGuard Integrated Inference Pipeline (Stage 3 + Stage 4)
=============================================================
Unified contract analysis engine combining:
1. Document chunking & offset tracking (Stage 2)
2. Multi-label chunk classification & gating (Stage 3)
3. Token-level span extraction & cross-chunk merging (Stage 4)

Produces structured clause results adhering to the Stage 5 contract.
"""

from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import json
import torch
import numpy as np
from transformers import AutoTokenizer

from ml.preprocessing.constants import LABEL_NAMES, NUM_LABELS, LABEL_TO_IDX, IDX_TO_LABEL
from ml.preprocessing.chunking import extract_natural_paragraphs, chunk_contract
from ml.inference.load_model import ClauseGuardInference
from ml.extraction.model import ClauseGuardSpanExtractor
from ml.extraction.token_tagging import decode_spans_from_bio_tags, merge_overlapping_clauses

DEFAULT_CLASSIFIER_PATH = "ml/checkpoints/roberta/best_checkpoint"
DEFAULT_EXTRACTOR_PATH = "ml/checkpoints/span_extractor/best_checkpoint"

class ClauseGuardPipeline:
    """
    End-to-end legal document analyzer supporting independent and classifier-assisted extraction.
    """

    def __init__(
        self,
        classifier_path: Union[str, Path] = DEFAULT_CLASSIFIER_PATH,
        extractor_path: Union[str, Path] = DEFAULT_EXTRACTOR_PATH,
        strategy: str = "assisted",  # 'assisted' or 'independent'
        assist_threshold: float = 0.20,
        min_span_confidence: float = 0.20,
        device: Optional[str] = None
    ):
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.strategy = strategy.lower()
        self.assist_threshold = assist_threshold
        self.min_span_confidence = min_span_confidence

        if self.strategy not in ["assisted", "independent"]:
            raise ValueError(f"Unsupported strategy '{self.strategy}'. Use 'assisted' or 'independent'.")

        self.classifier_path = Path(classifier_path)
        self.extractor_path = Path(extractor_path)

        # 1. Load Stage 3 Classifier
        print(f"[ClauseGuardPipeline] Loading Stage 3 Classifier from {self.classifier_path}...")
        self.classifier = ClauseGuardInference(self.classifier_path, device=self.device)

        # 2. Load Stage 4 Extractor
        print(f"[ClauseGuardPipeline] Loading Stage 4 Span Extractor from {self.extractor_path}...")
        self.extractor, self.extractor_meta = ClauseGuardSpanExtractor.from_pretrained(
            self.extractor_path,
            device=self.device
        )
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.extractor_path))
        self.model_version = self.extractor_meta.get("model_type", "roberta-span-v1")

    def analyze_chunk(
        self,
        chunk_text: str,
        chunk_id: str = "chunk_0",
        chunk_char_start: int = 0,
        document_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Analyzes a single chunk of text and extracts localized clause spans.
        """
        # Step 1: Run Stage 3 Classification
        clf_result = self.classifier.predict(chunk_text, max_length=256)
        cat_probs = {cat: clf_result["categories"][cat]["probability"] for cat in LABEL_NAMES}

        # Step 2: Determine which channels to extract based on strategy
        channel_whitelist = None
        if self.strategy == "assisted":
            channel_whitelist = [
                LABEL_TO_IDX[cat] for cat in LABEL_NAMES
                if cat_probs[cat] >= self.assist_threshold or clf_result["categories"][cat]["detected"]
            ]
            # If no category meets assistance threshold, skip extraction
            if not channel_whitelist:
                return []

        # Step 3: Run Stage 4 Span Extractor
        encoding = self.tokenizer(
            chunk_text,
            max_length=384,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            logits = self.extractor(input_ids, attention_mask)
            probs, tags = self.extractor.predict_tags_and_probs(logits)

        probs_np = probs.squeeze(0).cpu().numpy()  # [L, 9, 3]
        tags_np = tags.squeeze(0).cpu().numpy()    # [L, 9]
        offsets_np = encoding["offset_mapping"].squeeze(0).numpy()

        extracted = decode_spans_from_bio_tags(
            tag_predictions=tags_np,
            tag_probabilities=probs_np,
            offset_mapping=offsets_np,
            chunk_text=chunk_text,
            chunk_char_start=chunk_char_start,
            chunk_id=chunk_id,
            document_id=document_id,
            min_confidence=self.min_span_confidence,
            model_version=self.model_version,
            channel_whitelist=channel_whitelist
        )

        return extracted

    def analyze_document(
        self,
        document_text: str,
        document_id: Optional[str] = "contract_doc",
        target_tokens_per_chunk: int = 300,
        overlap_sentences: int = 2
    ) -> Dict[str, Any]:
        """
        Executes end-to-end document analysis:
        1. Natural paragraph chunking with character offset tracking.
        2. Chunk-level classification and span extraction.
        3. Cross-chunk boundary resolution and span deduplication.
        4. Structured results complying with the Stage 5 contract.
        """
        # Chunk contract text
        chunks = chunk_contract(
            contract_id=document_id or "doc",
            context=document_text
        )

        all_extracted_spans = []

        for ch in chunks:
            ch_spans = self.analyze_chunk(
                chunk_text=ch.text,
                chunk_id=ch.chunk_id,
                chunk_char_start=ch.char_start,
                document_id=document_id
            )
            all_extracted_spans.extend(ch_spans)

        # Merge cross-chunk and overlapping spans across the complete contract
        merged_clauses = merge_overlapping_clauses(
            clauses=all_extracted_spans,
            full_document_text=document_text
        )

        return {
            "document_id": document_id,
            "document_char_length": len(document_text),
            "total_chunks": len(chunks),
            "total_raw_detections": len(all_extracted_spans),
            "total_extracted_clauses": len(merged_clauses),
            "strategy": self.strategy,
            "assist_threshold": self.assist_threshold if self.strategy == "assisted" else None,
            "pipeline_version": "clauseguard-stage4-v1",
            "clauses": merged_clauses
        }
