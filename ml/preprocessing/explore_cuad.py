"""
ClauseGuard - CUAD Dataset Reconnaissance & Exploration Script
==============================================================
Reproducible script for analyzing the Contract Understanding Atticus Dataset (CUAD) v1.
Computes dataset statistics, validates character offset reconstruction, evaluates
class imbalance, category coverage, span lengths, document lengths, and multi-label
overlap patterns.

Outputs:
- docs/dataset_statistics.json
- docs/category_summary.csv
"""

import os
import sys
import json
import re
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import pandas as pd

RAW_DATA_DIR = Path("ml/data/raw/cuad")
DOCS_DIR = Path("docs")

def compute_percentiles(values):
    if not values:
        return {
            "min": 0, "p25": 0, "median": 0, "mean": 0.0,
            "p75": 0, "p90": 0, "max": 0, "std": 0.0
        }
    arr = np.array(values, dtype=float)
    return {
        "min": float(np.min(arr)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
        "std": float(np.std(arr))
    }

def run_exploration():
    print("=" * 80)
    print("CLAUSEGUARD | CUAD DATASET RECONNAISSANCE & EXPLORATION")
    print("=" * 80)

    # Validate raw data exists
    cuad_v1_path = RAW_DATA_DIR / "CUADv1.json"
    train_path = RAW_DATA_DIR / "train_separate_questions.json"
    test_path = RAW_DATA_DIR / "test.json"
    cat_desc_path = RAW_DATA_DIR / "category_descriptions.csv"

    for p in [cuad_v1_path, train_path, test_path, cat_desc_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required raw dataset file not found: {p}. Run download_cuad.py first.")

    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load category descriptions and build canonical normalized lookup
    print("\n[1/6] Loading category metadata and aligning schemas...")
    cat_desc_df = pd.read_csv(cat_desc_path)
    # Map normalized lower-case name to metadata
    category_meta = {}
    for _, row in cat_desc_df.iterrows():
        raw_cat = str(row.iloc[0]).replace("Category: ", "").strip()
        desc = str(row.iloc[1]).replace("Description: ", "").strip()
        ans_fmt = str(row.iloc[2]).replace("Answer Format: ", "").strip()
        grp = str(row.iloc[3]).replace("Group: ", "").strip()
        category_meta[raw_cat.lower()] = {
            "canonical_name": raw_cat,
            "description": desc,
            "answer_format": ans_fmt,
            "group": grp
        }
    print(f"Loaded metadata for {len(category_meta)} categories from category_descriptions.csv.")

    # 2. Load JSON files
    print("\n[2/6] Loading and parsing CUAD JSON files...")
    with open(cuad_v1_path, "r", encoding="utf-8") as f:
        cuad_full = json.load(f)
    with open(train_path, "r", encoding="utf-8") as f:
        cuad_train = json.load(f)
    with open(test_path, "r", encoding="utf-8") as f:
        cuad_test = json.load(f)

    full_contracts = cuad_full["data"]
    train_contracts = cuad_train["data"]
    test_contracts = cuad_test["data"]

    train_titles = set(c["title"] for c in train_contracts)
    test_titles = set(c["title"] for c in test_contracts)
    contract_overlap = train_titles.intersection(test_titles)

    print(f"Full dataset contracts: {len(full_contracts)}")
    print(f"Train contracts: {len(train_contracts)}")
    print(f"Test contracts: {len(test_contracts)}")
    print(f"Train/Test title intersection: {len(contract_overlap)} (Leakage check: {'PASS' if len(contract_overlap) == 0 else 'FAIL'})")

    # 3. Document and Paragraph Length Analysis & Offset Validation
    print("\n[3/6] Performing offset validation and document length analysis...")
    doc_char_lengths = []
    doc_word_lengths = []
    doc_para_counts = []
    para_char_lengths = []
    para_word_lengths = []

    total_annotations = 0
    exact_offset_matches = 0
    offset_mismatches = 0
    mismatch_samples = []

    spans_fully_in_para = 0
    spans_cross_para = 0

    # Per-category accumulators keyed by canonical name
    cat_spans = defaultdict(list)
    cat_span_char_lens = defaultdict(list)
    cat_span_word_lens = defaultdict(list)
    cat_contracts_pos = defaultdict(set)
    cat_train_contracts_pos = defaultdict(set)
    cat_test_contracts_pos = defaultdict(set)
    cat_train_spans_count = defaultdict(int)
    cat_test_spans_count = defaultdict(int)

    all_spans_by_contract = defaultdict(list)  # title -> list of (start, end, canonical_cat, text)

    for contract in full_contracts:
        title = contract["title"]
        full_context = contract["paragraphs"][0]["context"]
        doc_char_lengths.append(len(full_context))
        doc_word_lengths.append(len(full_context.split()))

        # Natural paragraph splitting
        para_matches = list(re.finditer(r"\S.*?(?=\n\s*\n|\Z)", full_context, re.DOTALL))
        if len(para_matches) <= 1:
            para_matches = list(re.finditer(r"\S.*?(?=\n|\Z)", full_context, re.DOTALL))
        
        doc_para_counts.append(len(para_matches))
        for pm in para_matches:
            ptxt = pm.group()
            para_char_lengths.append(len(ptxt))
            para_word_lengths.append(len(ptxt.split()))

        # Process QAs
        for qa in contract["paragraphs"][0]["qas"]:
            raw_cat_name = qa["id"].split("__")[-1]
            meta = category_meta.get(raw_cat_name.lower())
            if meta:
                canonical_cat = meta["canonical_name"]
            else:
                canonical_cat = raw_cat_name

            answers = qa.get("answers", [])
            if answers:
                cat_contracts_pos[canonical_cat].add(title)
                if title in train_titles:
                    cat_train_contracts_pos[canonical_cat].add(title)
                    cat_train_spans_count[canonical_cat] += len(answers)
                if title in test_titles:
                    cat_test_contracts_pos[canonical_cat].add(title)
                    cat_test_spans_count[canonical_cat] += len(answers)

                for ans in answers:
                    total_annotations += 1
                    s_idx = ans["answer_start"]
                    span_txt = ans["text"]
                    e_idx = s_idx + len(span_txt)

                    # Validation
                    actual_slice = full_context[s_idx:e_idx]
                    if actual_slice == span_txt:
                        exact_offset_matches += 1
                    else:
                        offset_mismatches += 1
                        if len(mismatch_samples) < 5:
                            mismatch_samples.append({
                                "contract": title,
                                "category": canonical_cat,
                                "start": s_idx,
                                "expected": span_txt[:40],
                                "actual": actual_slice[:40]
                            })

                    # Boundary containment
                    is_contained = any(pm.start() <= s_idx and e_idx <= pm.end() for pm in para_matches)
                    if is_contained:
                        spans_fully_in_para += 1
                    else:
                        spans_cross_para += 1

                    # Accumulate
                    cat_spans[canonical_cat].append(span_txt)
                    cat_span_char_lens[canonical_cat].append(len(span_txt))
                    cat_span_word_lens[canonical_cat].append(len(span_txt.split()))
                    all_spans_by_contract[title].append((s_idx, e_idx, canonical_cat, span_txt))

    print(f"Total annotations verified: {total_annotations:,}")
    print(f"Exact offset matches: {exact_offset_matches:,} ({exact_offset_matches/total_annotations*100:.2f}%)")
    print(f"Offset mismatches: {offset_mismatches:,}")
    print(f"Spans fully inside natural paragraph: {spans_fully_in_para:,} ({spans_fully_in_para/total_annotations*100:.2f}%)")
    print(f"Spans crossing paragraph boundary: {spans_cross_para:,} ({spans_cross_para/total_annotations*100:.2f}%)")

    # 4. Multi-label Overlap Analysis
    print("\n[4/6] Analyzing multi-label span overlap and co-occurrence...")
    intra_cat_overlap = 0
    inter_cat_overlap = 0
    pairwise_cooccurrences = Counter()
    paras_with_multi_cats = 0
    total_eval_paras = 0

    for contract in full_contracts:
        title = contract["title"]
        full_context = contract["paragraphs"][0]["context"]
        spans = all_spans_by_contract[title]

        # Check pairwise span overlaps
        for i in range(len(spans)):
            s1, e1, c1, _ = spans[i]
            for j in range(i + 1, len(spans)):
                s2, e2, c2, _ = spans[j]
                if max(s1, s2) < min(e1, e2):  # Overlap condition
                    if c1 == c2:
                        intra_cat_overlap += 1
                    else:
                        inter_cat_overlap += 1
                        pair = tuple(sorted([c1, c2]))
                        pairwise_cooccurrences[pair] += 1

        # Check multi-category co-occurrence inside natural paragraphs
        para_matches = list(re.finditer(r"\S.*?(?=\n\s*\n|\Z)", full_context, re.DOTALL))
        if len(para_matches) <= 1:
            para_matches = list(re.finditer(r"\S.*?(?=\n|\Z)", full_context, re.DOTALL))

        for pm in para_matches:
            total_eval_paras += 1
            p_start, p_end = pm.start(), pm.end()
            cats_in_para = set()
            for s, e, cat, _ in spans:
                if max(s, p_start) < min(e, p_end):
                    cats_in_para.add(cat)
            if len(cats_in_para) > 1:
                paras_with_multi_cats += 1

    print(f"Intra-category span overlaps: {intra_cat_overlap:,}")
    print(f"Inter-category span overlaps: {inter_cat_overlap:,}")
    print(f"Paragraphs containing multiple categories: {paras_with_multi_cats:,} of {total_eval_paras:,} ({paras_with_multi_cats/total_eval_paras*100:.2f}%)")

    # 5. Build category statistics table
    print("\n[5/6] Compiling category statistics...")
    category_summary_rows = []
    category_stats_dict = {}

    canonical_list = sorted([m["canonical_name"] for m in category_meta.values()])

    for cat in canonical_list:
        meta = category_meta.get(cat.lower(), {"description": "N/A", "answer_format": "N/A", "group": "N/A"})
        spans = cat_spans.get(cat, [])
        pos_contracts_cnt = len(cat_contracts_pos.get(cat, set()))
        train_pos_cnt = len(cat_train_contracts_pos.get(cat, set()))
        test_pos_cnt = len(cat_test_contracts_pos.get(cat, set()))
        total_spans_cnt = len(spans)
        train_spans_cnt = cat_train_spans_count.get(cat, 0)
        test_spans_cnt = cat_test_spans_count.get(cat, 0)

        char_stats = compute_percentiles(cat_span_char_lens.get(cat, []))
        word_stats = compute_percentiles(cat_span_word_lens.get(cat, []))

        category_stats_dict[cat] = {
            "description": meta["description"],
            "answer_format": meta["answer_format"],
            "atticus_group": meta["group"],
            "total_spans": total_spans_cnt,
            "train_spans": train_spans_cnt,
            "test_spans": test_spans_cnt,
            "positive_contracts": pos_contracts_cnt,
            "positive_contracts_pct": round(pos_contracts_cnt / 510 * 100, 2),
            "train_positive_contracts": train_pos_cnt,
            "train_positive_contracts_pct": round(train_pos_cnt / 408 * 100, 2),
            "test_positive_contracts": test_pos_cnt,
            "test_positive_contracts_pct": round(test_pos_cnt / 102 * 100, 2),
            "avg_spans_per_positive_contract": round(total_spans_cnt / pos_contracts_cnt, 2) if pos_contracts_cnt > 0 else 0,
            "span_char_length": char_stats,
            "span_word_length": word_stats
        }

        category_summary_rows.append({
            "category": cat,
            "group": meta["group"],
            "answer_format": meta["answer_format"],
            "positive_contracts": pos_contracts_cnt,
            "contract_coverage_pct": round(pos_contracts_cnt / 510 * 100, 1),
            "total_spans": total_spans_cnt,
            "train_spans": train_spans_cnt,
            "test_spans": test_spans_cnt,
            "avg_spans_per_pos_contract": round(total_spans_cnt / pos_contracts_cnt, 2) if pos_contracts_cnt > 0 else 0,
            "median_chars": int(char_stats["median"]),
            "median_words": int(word_stats["median"]),
            "max_words": int(word_stats["max"])
        })

    # Sort summary dataframe by total spans descending
    cat_summary_df = pd.DataFrame(category_summary_rows).sort_values(by="total_spans", ascending=False)
    csv_out_path = DOCS_DIR / "category_summary.csv"
    cat_summary_df.to_csv(csv_out_path, index=False)
    print(f"Saved category summary CSV to {csv_out_path}")

    # Top overlapping pairs
    top_overlaps = [
        {"category_a": p[0], "category_b": p[1], "overlap_count": count}
        for p, count in pairwise_cooccurrences.most_common(20)
    ]

    # 6. Overall Dataset Statistics Object
    print("\n[6/6] Generating comprehensive JSON statistics...")
    dataset_statistics = {
        "dataset_name": "Contract Understanding Atticus Dataset (CUAD) v1",
        "provenance": {
            "authors": "Dan Hendrycks, Collin Burns, Anya Chen, Spencer Ball, et al. (The Atticus Project)",
            "paper": "CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review (NeurIPS 2021)",
            "license": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
            "source_corpus": "510 commercial legal contracts from US SEC EDGAR filings across 25 contract types"
        },
        "high_level_counts": {
            "total_contracts": len(full_contracts),
            "total_benchmark_train_contracts": len(train_contracts),
            "total_benchmark_test_contracts": len(test_contracts),
            "contract_leakage_overlap": len(contract_overlap),
            "total_categories": len(category_meta),
            "total_qa_queries": len(full_contracts) * len(category_meta),
            "positive_qa_queries": sum(len(cat_contracts_pos[c]) for c in canonical_list),
            "negative_qa_queries": (len(full_contracts) * len(category_meta)) - sum(len(cat_contracts_pos[c]) for c in canonical_list),
            "overall_positive_query_ratio": round(sum(len(cat_contracts_pos[c]) for c in canonical_list) / (len(full_contracts) * len(category_meta)), 4),
            "total_annotation_spans": total_annotations,
            "train_annotation_spans": sum(cat_train_spans_count.values()),
            "test_annotation_spans": sum(cat_test_spans_count.values()),
            "total_natural_paragraphs": len(para_char_lengths)
        },
        "validation_metrics": {
            "total_annotations_checked": total_annotations,
            "exact_offset_matches": exact_offset_matches,
            "offset_mismatches": offset_mismatches,
            "offset_match_accuracy_pct": round(exact_offset_matches / total_annotations * 100, 4),
            "spans_fully_within_natural_paragraph": spans_fully_in_para,
            "spans_fully_within_paragraph_pct": round(spans_fully_in_para / total_annotations * 100, 2),
            "spans_crossing_paragraph_boundaries": spans_cross_para,
            "spans_crossing_paragraph_pct": round(spans_cross_para / total_annotations * 100, 2)
        },
        "multi_label_overlap_metrics": {
            "intra_category_span_overlaps": intra_cat_overlap,
            "inter_category_span_overlaps": inter_cat_overlap,
            "paragraphs_with_multiple_categories": paras_with_multi_cats,
            "paragraphs_with_multiple_categories_pct": round(paras_with_multi_cats / total_eval_paras * 100, 2),
            "top_overlapping_category_pairs": top_overlaps
        },
        "contract_length_distribution": {
            "characters": compute_percentiles(doc_char_lengths),
            "words": compute_percentiles(doc_word_lengths),
            "paragraphs_per_contract": compute_percentiles(doc_para_counts)
        },
        "paragraph_length_distribution": {
            "characters": compute_percentiles(para_char_lengths),
            "words": compute_percentiles(para_word_lengths),
            "paragraphs_exceeding_400_words": int(sum(1 for w in para_word_lengths if w > 400)),
            "paragraphs_exceeding_400_words_pct": round(sum(1 for w in para_word_lengths if w > 400) / len(para_word_lengths) * 100, 2)
        },
        "category_statistics": category_stats_dict
    }

    json_out_path = DOCS_DIR / "dataset_statistics.json"
    with open(json_out_path, "w", encoding="utf-8") as f:
        json.dump(dataset_statistics, f, indent=2)
    print(f"Saved machine-generated dataset statistics to {json_out_path}")

    # Display Top 10 and Bottom 10 categories
    print("\n" + "=" * 90)
    print("CATEGORY DISTRIBUTION SUMMARY (ALL 41 CATEGORIES RANKED BY SPAN COUNT)")
    print("=" * 90)
    print(f"{'#':<3} | {'Category':<35} | {'Cov%':<6} | {'Spans':<6} | {'Avg/Pos':<8} | {'MedWords':<8} | {'MaxWords'}")
    print("-" * 90)
    for idx, (_, r) in enumerate(cat_summary_df.iterrows(), 1):
        print(f"{idx:<3} | {r['category']:<35} | {r['contract_coverage_pct']:<6.1f} | {r['total_spans']:<6} | {r['avg_spans_per_pos_contract']:<8.2f} | {r['median_words']:<8} | {r['max_words']}")
    print("=" * 90)
    print("Exploration completed successfully.")

if __name__ == "__main__":
    run_exploration()
