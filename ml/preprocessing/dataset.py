"""
ClauseGuard PyTorch Dataset for Multi-Label Legal Clause Classification
========================================================================
Implements a reusable, tokenizer-independent PyTorch Dataset:
- Loads processed Parquet chunks (train, val, test).
- Supports dynamic runtime tokenization for any Hugging Face model (RoBERTa, InLegalBERT, etc.).
- Returns multi-label targets (shape [9]) and supervision masks (shape [9]).
- Preserves token offset mappings and global character offsets for document highlighting.
"""

from typing import Union, Optional, Dict, Any, List, Tuple
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset
from ml.preprocessing.constants import LABEL_NAMES, NUM_LABELS

class ClauseGuardDataset(Dataset):
    """
    PyTorch Dataset for ClauseGuard multi-label legal text chunks.
    Compatible with any Hugging Face PreTrainedTokenizerFast or PreTrainedTokenizer.
    """

    def __init__(
        self,
        data_source: Union[str, Path, pd.DataFrame],
        tokenizer: Optional[Any] = None,
        max_length: int = 512,
        return_metadata: bool = True
    ):
        """
        Args:
            data_source: Path to parquet file or pre-loaded DataFrame.
            tokenizer: Hugging Face tokenizer instance (optional). If None, returns raw text.
            max_length: Maximum sequence length (default 512 for transformer models).
            return_metadata: Whether to include offset/document metadata in getitem.
        """
        if isinstance(data_source, (str, Path)):
            self.df = pd.read_parquet(data_source)
        elif isinstance(data_source, pd.DataFrame):
            self.df = data_source
        else:
            raise TypeError(f"Unsupported data_source type: {type(data_source)}")

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.return_metadata = return_metadata

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]

        text = str(row["text"])
        labels = torch.tensor(list(row["labels"]), dtype=torch.float32)
        label_mask = torch.tensor(list(row["label_mask"]), dtype=torch.float32)

        item: Dict[str, Any] = {
            "chunk_id": str(row["chunk_id"]),
            "contract_id": str(row["contract_id"]),
            "labels": labels,
            "label_mask": label_mask,
            "char_start": int(row["char_start"]),
            "char_end": int(row["char_end"]),
        }

        if self.tokenizer is not None:
            # Tokenize dynamically with offset mapping
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                return_offsets_mapping=getattr(self.tokenizer, "is_fast", False)
            )

            item["input_ids"] = encoding["input_ids"].squeeze(0)
            item["attention_mask"] = encoding["attention_mask"].squeeze(0)

            if "token_type_ids" in encoding:
                item["token_type_ids"] = encoding["token_type_ids"].squeeze(0)

            if "offset_mapping" in encoding:
                item["offset_mapping"] = encoding["offset_mapping"].squeeze(0)
        else:
            item["text"] = text

        if self.return_metadata:
            item["metadata"] = {
                "chunk_index": int(row["chunk_index"]),
                "split": str(row.get("split", "")),
                "word_count": int(row["word_count"]),
                "approx_tokens": int(row["approx_tokens"]),
                "label_names": list(row.get("label_names", LABEL_NAMES)),
                "spans": list(row.get("spans", [])),
                "span_categories": list(row.get("span_categories", [])),
                "is_positive": bool(row.get("is_positive", False))
            }

        return item

def project_token_span_to_global(
    chunk_char_start: int,
    token_start_idx: int,
    token_end_idx: int,
    offset_mapping: torch.Tensor
) -> Tuple[int, int]:
    """
    Converts model token index span [token_start_idx, token_end_idx] back to
    global contract character offsets [global_start, global_end].

    Args:
        chunk_char_start: Starting global character position of the chunk in the contract.
        token_start_idx: Index of starting token in sequence.
        token_end_idx: Index of ending token in sequence (inclusive).
        offset_mapping: Tensor of shape [seq_len, 2] from fast tokenizer.

    Returns:
        (global_char_start, global_char_end) in the original raw contract text.
    """
    local_start = int(offset_mapping[token_start_idx][0])
    local_end = int(offset_mapping[token_end_idx][1])
    global_start = chunk_char_start + local_start
    global_end = chunk_char_start + local_end
    return global_start, global_end

def clauseguard_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for PyTorch DataLoader.
    Batches tensors while preserving metadata lists.
    """
    collated: Dict[str, Any] = {}

    keys = batch[0].keys()
    for k in keys:
        vals = [b[k] for b in batch]
        if isinstance(vals[0], torch.Tensor):
            collated[k] = torch.stack(vals, dim=0)
        else:
            collated[k] = vals

    return collated
