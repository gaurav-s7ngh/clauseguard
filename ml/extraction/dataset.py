"""
ClauseGuard Span Extraction Dataset & DataLoader Engine
=======================================================
Efficient pre-tokenized tensor dataset for multi-channel token classification:
- Converts Parquet chunk records and spans into 9-channel BIO tensor targets.
- Preserves token character offsets for exact span reconstruction.
- Zero-heap-fragmentation tensor architecture for high-throughput CPU training.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast

from ml.preprocessing.constants import NUM_LABELS, RANDOM_SEED
from ml.extraction.token_tagging import align_spans_to_bio_labels

class PretokenizedSpanDataset(Dataset):
    """
    In-memory tensor dataset for token classification with 9 BIO channels.
    """

    def __init__(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        offset_mapping: Optional[torch.Tensor] = None,
        chunk_ids: Optional[List[str]] = None,
        char_starts: Optional[List[int]] = None,
        texts: Optional[List[str]] = None,
        token_type_ids: Optional[torch.Tensor] = None
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels
        self.offset_mapping = offset_mapping
        self.chunk_ids = chunk_ids
        self.char_starts = char_starts
        self.texts = texts
        self.token_type_ids = token_type_ids

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx]
        }
        if self.token_type_ids is not None:
            item["token_type_ids"] = self.token_type_ids[idx]
        if self.offset_mapping is not None:
            item["offset_mapping"] = self.offset_mapping[idx]
        if self.chunk_ids is not None:
            item["chunk_id"] = self.chunk_ids[idx]
        if self.char_starts is not None:
            item["char_start"] = self.char_starts[idx]
        if self.texts is not None:
            item["text"] = self.texts[idx]
        return item

def build_span_dataset(
    df: pd.DataFrame,
    tokenizer: PreTrainedTokenizerFast,
    max_length: int = 384,
    include_metadata: bool = True
) -> PretokenizedSpanDataset:
    """
    Builds a PretokenizedSpanDataset from a DataFrame of chunk records.
    """
    input_ids_list = []
    attention_mask_list = []
    labels_list = []
    offsets_list = []

    texts = df["text"].tolist()
    spans_col = df["spans"].tolist()
    masks_col = df["label_mask"].tolist()
    chunk_ids = df["chunk_id"].tolist() if include_metadata else None
    char_starts = df["char_start"].tolist() if include_metadata else None

    for idx in range(len(df)):
        text = texts[idx]
        spans = spans_col[idx]
        label_mask = masks_col[idx]

        enc, label_matrix = align_spans_to_bio_labels(
            text=text,
            spans=spans,
            label_mask=label_mask,
            tokenizer=tokenizer,
            max_length=max_length
        )

        input_ids_list.append(enc["input_ids"][0])
        attention_mask_list.append(enc["attention_mask"][0])
        labels_list.append(torch.tensor(label_matrix, dtype=torch.long))

        if include_metadata:
            offsets_list.append(enc["offset_mapping"][0])

    all_input_ids = torch.stack(input_ids_list)
    all_attention_mask = torch.stack(attention_mask_list)
    all_labels = torch.stack(labels_list)
    all_offsets = torch.stack(offsets_list) if include_metadata else None

    del input_ids_list, attention_mask_list, labels_list, offsets_list
    import gc; gc.collect()

    return PretokenizedSpanDataset(
        input_ids=all_input_ids,
        attention_mask=all_attention_mask,
        labels=all_labels,
        offset_mapping=all_offsets,
        chunk_ids=chunk_ids,
        char_starts=char_starts,
        texts=texts if include_metadata else None
    )

def prepare_span_training_df(
    parquet_path: str,
    max_samples: Optional[int] = 1000,
    pos_ratio: float = 0.80,
    seed: int = RANDOM_SEED
) -> pd.DataFrame:
    """
    Loads chunks for span extraction training:
    - Focuses primarily on positive chunks (80% positives, 20% negatives).
    - Guarantees representation across all 9 categories.
    """
    cols = ["chunk_id", "contract_id", "text", "labels", "label_mask", "char_start", "char_end", "is_positive", "spans", "span_categories"]
    df = pd.read_parquet(parquet_path, columns=cols)

    if max_samples is None or len(df) <= max_samples:
        return df

    pos_df = df[df["is_positive"] == True]
    neg_df = df[df["is_positive"] == False]

    n_pos = min(int(max_samples * pos_ratio), len(pos_df))
    n_neg = max_samples - n_pos

    sampled_pos = pos_df.sample(n=n_pos, random_state=seed)
    sampled_neg = neg_df.sample(n=min(n_neg, len(neg_df)), random_state=seed)

    combined = pd.concat([sampled_pos, sampled_neg]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    del df, pos_df, neg_df, sampled_pos, sampled_neg
    import gc; gc.collect()

    return combined
