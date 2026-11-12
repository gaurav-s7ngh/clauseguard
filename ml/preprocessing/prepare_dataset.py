"""
ClauseGuard Dataset Preparation & Preprocessing Pipeline
=========================================================
Main entry point for Stage 2 preprocessing:
- Ingests raw CUAD v1 dataset from ml/data/raw/cuad/
- Applies deterministic contract-level splitting (340 train, 68 val, 102 test)
- Performs paragraph-aware legal chunking with span lookahead
- Generates 9-label multi-hot targets and supervision masks
- Validates 100% exact character offset reconstruction
- Exports processed Parquet datasets to ml/data/processed/
- Exports machine-readable validation metrics to docs/preprocessing_validation_report.json

Usage:
    python ml/preprocessing/prepare_dataset.py
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict
from dataclasses import asdict

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Add workspace root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ml.preprocessing.constants import (
    RANDOM_SEED,
    LABEL_NAMES,
    LABEL_POSITIVE,
    LABEL_NEGATIVE,
    LABEL_UNKNOWN,
    MASK_SUPERVISED,
    MASK_UNSUPERVISED,
)
from ml.preprocessing.splitter import create_contract_splits
from ml.preprocessing.chunking import chunk_contract, TextSpan
from ml.preprocessing.labeling import (
    parse_contract_annotations,
    label_chunk,
    LabeledChunk,
)

RAW_DATA_DIR = Path("ml/data/raw/cuad")
PROCESSED_DATA_DIR = Path("ml/data/processed")
DOCS_DIR = Path("docs")

def run_pipeline() -> Dict[str, Any]:
    print("=" * 80)
    print("CLAUSEGUARD | DATASET PREPROCESSING & TRAINING PIPELINE (STAGE 2)")
    print("=" * 80)

    # 1. Verification of inputs
    cuad_full_path = RAW_DATA_DIR / "CUADv1.json"
    train_split_path = RAW_DATA_DIR / "train_separate_questions.json"
    test_split_path = RAW_DATA_DIR / "test.json"

    for p in [cuad_full_path, train_split_path, test_split_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required raw CUAD file: {p}. Ensure Stage 1 download succeeded.")

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1/5] Loading raw CUAD datasets...")
    with open(cuad_full_path, "r", encoding="utf-8") as f:
        cuad_full = json.load(f)
    with open(train_split_path, "r", encoding="utf-8") as f:
        cuad_train = json.load(f)
    with open(test_split_path, "r", encoding="utf-8") as f:
        cuad_test = json.load(f)

    official_train_titles = sorted(list(set(c["title"] for c in cuad_train["data"])))
    official_test_titles = sorted(list(set(c["title"] for c in cuad_test["data"])))

    # 2. Contract-level splitting
    print("\n[2/5] Creating deterministic contract-level splits...")
    split_map = create_contract_splits(
        official_train_titles=official_train_titles,
        official_test_titles=official_test_titles,
        random_seed=RANDOM_SEED
    )

    split_counts = Counter_map = defaultdict(int)
    for s in split_map.values():
        split_counts[s] += 1
    print(f"Contract splits established: Train={split_counts['train']}, Val={split_counts['val']}, Test={split_counts['test']}")

    # 3. Processing and Chunking
    print("\n[3/5] Chunking contracts and generating multi-label annotations...")
    chunks_by_split: Dict[str, List[Dict[str, Any]]] = {
        "train": [],
        "val": [],
        "test": []
    }

    total_annotations_checked = 0
    exact_offset_matches = 0
    all_validation_exceptions = []
    cross_chunk_spans_logged = []

    # Category tracking per split: split -> category -> {positive, negative, unknown}
    category_counts_per_split = {
        s: {
            cat: {"positive": 0, "verified_negative": 0, "unknown": 0}
            for cat in LABEL_NAMES
        }
        for s in ["train", "val", "test"]
    }

    contract_dict_by_title = {c["title"]: c for c in cuad_full["data"]}

    for c_idx, (title, split_name) in enumerate(split_map.items(), 1):
        contract_data = contract_dict_by_title[title]
        context = contract_data["paragraphs"][0]["context"]

        # Parse & validate contract annotations
        spans, cat_impossible, exceptions = parse_contract_annotations(contract_data, context)
        all_validation_exceptions.extend(exceptions)

        for sp in spans:
            total_annotations_checked += 1
            if context[sp.char_start:sp.char_end] == sp.text:
                exact_offset_matches += 1

        # Chunk contract with span lookahead
        chunks = chunk_contract(
            contract_id=title,
            context=context,
            known_spans=spans
        )

        # Label each chunk
        for ch in chunks:
            # Verify exact chunk text integrity against context
            assert context[ch.char_start:ch.char_end] == ch.text, f"Text mismatch in chunk {ch.chunk_id}"

            labeled = label_chunk(
                chunk=ch,
                contract_spans=spans,
                category_impossible=cat_impossible,
                split_name=split_name
            )

            # Record stats
            for idx, cat_name in enumerate(LABEL_NAMES):
                lbl_val = labeled.labels[idx]
                mask_val = labeled.label_mask[idx]
                if lbl_val == LABEL_POSITIVE:
                    category_counts_per_split[split_name][cat_name]["positive"] += 1
                elif mask_val == MASK_SUPERVISED:
                    category_counts_per_split[split_name][cat_name]["verified_negative"] += 1
                else:
                    category_counts_per_split[split_name][cat_name]["unknown"] += 1

            # Log any cross-chunk spans
            for sp_rec in labeled.spans:
                if sp_rec.get("is_cross_chunk", False):
                    cross_chunk_spans_logged.append({
                        "contract_id": title,
                        "chunk_id": labeled.chunk_id,
                        "category": sp_rec["category"],
                        "global_start": sp_rec["global_start"],
                        "global_end": sp_rec["global_end"],
                        "local_start": sp_rec["local_start"],
                        "local_end": sp_rec["local_end"]
                    })

            chunks_by_split[split_name].append(asdict(labeled))

        if c_idx % 100 == 0 or c_idx == len(split_map):
            print(f"  Processed {c_idx}/{len(split_map)} contracts...")

    print(f"Total annotations checked: {total_annotations_checked:,}")
    print(f"Exact offset matches: {exact_offset_matches:,} ({exact_offset_matches/total_annotations_checked*100:.2f}%)")
    print(f"Validation exceptions: {len(all_validation_exceptions)}")

    # 4. Export Parquet Datasets
    print("\n[4/5] Saving processed Parquet datasets to ml/data/processed/...")
    dataset_paths = {}
    split_summary = {}

    for s_name in ["train", "val", "test"]:
        records = chunks_by_split[s_name]
        df = pd.DataFrame(records)
        out_path = PROCESSED_DATA_DIR / f"{s_name}_chunks.parquet"
        table = pa.Table.from_pandas(df)
        pq.write_table(table, out_path)
        dataset_paths[s_name] = str(out_path)

        pos_chunks = sum(1 for r in records if r["is_positive"])
        neg_chunks = len(records) - pos_chunks
        split_summary[s_name] = {
            "contracts": split_counts[s_name],
            "total_chunks": len(records),
            "positive_chunks": pos_chunks,
            "negative_chunks": neg_chunks,
            "positive_ratio": round(pos_chunks / len(records), 4) if records else 0,
            "file_size_bytes": out_path.stat().st_size
        }
        print(f"  ✓ {s_name}_chunks.parquet: {len(records):,} chunks ({pos_chunks:,} positive, {neg_chunks:,} negative) | {out_path.stat().st_size:,} bytes")

    # 5. Build and save comprehensive validation report
    print("\n[5/5] Generating machine-readable validation report...")
    validation_report = {
        "pipeline_name": "ClauseGuard Stage 2 Preprocessing Pipeline",
        "random_seed": RANDOM_SEED,
        "label_names": LABEL_NAMES,
        "split_summary": split_summary,
        "offset_validation": {
            "total_annotations_checked": total_annotations_checked,
            "exact_offset_matches": exact_offset_matches,
            "mismatch_count": total_annotations_checked - exact_offset_matches,
            "accuracy_pct": round(exact_offset_matches / total_annotations_checked * 100, 4) if total_annotations_checked else 100.0,
            "exceptions_logged": len(all_validation_exceptions)
        },
        "cross_chunk_spans": {
            "total_logged": len(cross_chunk_spans_logged),
            "description": "Spans spanning across paragraph/chunk boundaries; preserved in all overlapping chunks with local coordinates."
        },
        "per_category_distributions": category_counts_per_split
    }

    report_json_path = DOCS_DIR / "preprocessing_validation_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(validation_report, f, indent=2)
    print(f"Validation report saved to {report_json_path}")

    # Summary table
    print("\n" + "=" * 90)
    print("STAGE 2 DATASET GENERATION SUMMARY")
    print("=" * 90)
    print(f"{'Split':<8} | {'Contracts':<10} | {'Total Chunks':<14} | {'Pos Chunks':<12} | {'Neg Chunks':<12} | {'Pos %':<6}")
    print("-" * 90)
    for s_name in ["train", "val", "test"]:
        st = split_summary[s_name]
        print(f"{s_name:<8} | {st['contracts']:<10} | {st['total_chunks']:<14,d} | {st['positive_chunks']:<12,d} | {st['negative_chunks']:<12,d} | {st['positive_ratio']*100:<6.2f}%")
    print("=" * 90)

    print("\n" + "=" * 90)
    print("PER-CATEGORY POSITIVE / NEGATIVE / UNKNOWN COUNTS (TRAIN / VAL / TEST)")
    print("=" * 90)
    print(f"{'Category':<28} | {'Train (+/-/?)':<20} | {'Val (+/-/?)':<20} | {'Test (+/-/?)':<20}")
    print("-" * 90)
    for cat in LABEL_NAMES:
        tr_c = category_counts_per_split["train"][cat]
        va_c = category_counts_per_split["val"][cat]
        te_c = category_counts_per_split["test"][cat]
        tr_s = f"{tr_c['positive']}/{tr_c['verified_negative']}/{tr_c['unknown']}"
        va_s = f"{va_c['positive']}/{va_c['verified_negative']}/{va_c['unknown']}"
        te_s = f"{te_c['positive']}/{te_c['verified_negative']}/{te_c['unknown']}"
        print(f"{cat:<28} | {tr_s:<20} | {va_s:<20} | {te_s:<20}")
    print("=" * 90)

    return validation_report

if __name__ == "__main__":
    run_pipeline()
