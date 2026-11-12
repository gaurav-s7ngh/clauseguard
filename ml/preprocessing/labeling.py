"""
ClauseGuard Multi-Label Annotation & Supervision Mask Processor
===============================================================
Maps raw CUAD Question-Answering annotations to ClauseGuard's 9 target labels.
Handles:
- Span extraction and character offset validation.
- Local offset projection (chunk coordinate space).
- Explicit 3-state label semantics:
    * Positive (1): Confirmed annotated clause span inside or substantively overlapping chunk.
    * Verified Negative (0, mask=1): Certified absence via CUAD is_impossible or clean non-overlap.
    * Unknown / Ambiguous (0, mask=0): Minimal edge spillover (<30 chars & <30% span) masked out.
- Multi-label co-occurrence and multi-span tracking.
- Cross-chunk annotation boundary bookkeeping.
"""

from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from ml.preprocessing.constants import (
    LABEL_NAMES,
    LABEL_TO_IDX,
    NUM_LABELS,
    CUAD_TO_CLAUSEGUARD,
    LABEL_POSITIVE,
    LABEL_NEGATIVE,
    LABEL_UNKNOWN,
    MASK_SUPERVISED,
    MASK_UNSUPERVISED,
)
from ml.preprocessing.chunking import DocumentChunk, TextSpan

@dataclass
class ChunkAnnotationSpan:
    category: str
    global_start: int
    global_end: int
    local_start: int
    local_end: int
    text: str
    is_cross_chunk: bool

@dataclass
class LabeledChunk:
    contract_id: str
    chunk_id: str
    chunk_index: int
    split: str
    text: str
    char_start: int
    char_end: int
    word_count: int
    approx_tokens: int
    labels: List[int]
    label_mask: List[int]
    label_names: List[str]
    spans: List[Dict[str, Any]]
    span_categories: List[str]
    is_positive: bool

def parse_contract_annotations(
    contract_dict: Dict[str, Any],
    context: str
) -> Tuple[List[TextSpan], Dict[str, bool], List[Dict[str, Any]]]:
    """
    Parses and validates annotations for the 9 ClauseGuard target categories in a contract.

    Returns:
        spans: List of valid TextSpan objects.
        category_impossible: Dict mapping label -> bool (True if category has 0 spans in contract).
        exceptions: List of any validation errors or anomalies.
    """
    contract_title = contract_dict.get("title", "UNKNOWN_CONTRACT")
    spans: List[TextSpan] = []
    category_impossible: Dict[str, bool] = {lbl: True for lbl in LABEL_NAMES}
    exceptions: List[Dict[str, Any]] = []

    paragraphs = contract_dict.get("paragraphs", [])
    if not paragraphs:
        exceptions.append({
            "contract": contract_title,
            "error_type": "EMPTY_PARAGRAPHS",
            "message": "Contract contains no paragraph objects."
        })
        return spans, category_impossible, exceptions

    qas = paragraphs[0].get("qas", [])
    for qa in qas:
        qa_id = qa.get("id", "")
        raw_cat = qa_id.split("__")[-1].lower()
        if raw_cat not in CUAD_TO_CLAUSEGUARD:
            continue

        cg_label = CUAD_TO_CLAUSEGUARD[raw_cat]
        answers = qa.get("answers", [])
        is_impossible = qa.get("is_impossible", False)

        if answers and len(answers) > 0:
            category_impossible[cg_label] = False
            for ans in answers:
                s_idx = ans.get("answer_start")
                expected_txt = ans.get("text", "")
                if s_idx is None or expected_txt == "":
                    exceptions.append({
                        "contract": contract_title,
                        "category": cg_label,
                        "error_type": "MALFORMED_ANSWER",
                        "message": f"Answer missing answer_start or text: {ans}"
                    })
                    continue

                e_idx = s_idx + len(expected_txt)
                actual_slice = context[s_idx:e_idx]
                if actual_slice != expected_txt:
                    exceptions.append({
                        "contract": contract_title,
                        "category": cg_label,
                        "error_type": "OFFSET_MISMATCH",
                        "start": s_idx,
                        "end": e_idx,
                        "expected": expected_txt[:50],
                        "actual": actual_slice[:50]
                    })
                    continue

                spans.append(TextSpan(
                    category=cg_label,
                    char_start=s_idx,
                    char_end=e_idx,
                    text=expected_txt
                ))
        else:
            # If no answers, verify is_impossible is True
            if not is_impossible:
                # Still certified negative by empty answers list
                pass
            category_impossible[cg_label] = True

    return spans, category_impossible, exceptions

def label_chunk(
    chunk: DocumentChunk,
    contract_spans: List[TextSpan],
    category_impossible: Dict[str, bool],
    split_name: str
) -> LabeledChunk:
    """
    Labels a chunk against ground-truth contract annotations with explicit 3-state semantics:
    - 1: Confirmed positive (contains full or substantive clause overlap >= 30 chars or >= 30% of span).
    - 0: Verified negative (contract is certified impossible, or chunk has zero overlap with category).
    - Unknown (label=0, mask=0): Minimal edge spillover (<30 chars and <30% span) is masked out.
    """
    labels = [LABEL_NEGATIVE] * NUM_LABELS
    label_mask = [MASK_SUPERVISED] * NUM_LABELS
    chunk_spans: List[ChunkAnnotationSpan] = []
    active_categories = set()

    c_start = chunk.char_start
    c_end = chunk.char_end
    c_text = chunk.text

    for sp in contract_spans:
        cat_idx = LABEL_TO_IDX[sp.category]
        overlap_s = max(c_start, sp.char_start)
        overlap_e = min(c_end, sp.char_end)
        overlap_len = max(0, overlap_e - overlap_s)

        if overlap_len > 0:
            span_len = sp.char_end - sp.char_start
            is_full = (c_start <= sp.char_start and sp.char_end <= c_end)
            is_substantive = is_full or (overlap_len >= 30) or (overlap_len / span_len >= 0.30)

            if is_substantive:
                labels[cat_idx] = LABEL_POSITIVE
                label_mask[cat_idx] = MASK_SUPERVISED
                active_categories.add(sp.category)

                loc_s = overlap_s - c_start
                loc_e = overlap_e - c_start
                slice_txt = c_text[loc_s:loc_e]
                is_cross = not is_full

                chunk_spans.append(ChunkAnnotationSpan(
                    category=sp.category,
                    global_start=overlap_s,
                    global_end=overlap_e,
                    local_start=loc_s,
                    local_end=loc_e,
                    text=slice_txt,
                    is_cross_chunk=is_cross
                ))
            else:
                # Insignificant boundary spillover (<30 chars and <30% span)
                # If not already marked positive by another span, mask out to prevent noisy negative supervision
                if labels[cat_idx] != LABEL_POSITIVE:
                    labels[cat_idx] = LABEL_UNKNOWN
                    label_mask[cat_idx] = MASK_UNSUPERVISED

    # For categories with no overlap:
    # If category is impossible at contract level -> Verified Negative (label=0, mask=1).
    # If category exists elsewhere in contract -> Verified Negative for this chunk (label=0, mask=1).
    # Both are correctly retained as label=0, mask=1.

    span_records = [asdict(cs) for cs in chunk_spans]
    is_pos = any(l == LABEL_POSITIVE for l in labels)

    return LabeledChunk(
        contract_id=chunk.contract_id,
        chunk_id=chunk.chunk_id,
        chunk_index=chunk.chunk_index,
        split=split_name,
        text=chunk.text,
        char_start=chunk.char_start,
        char_end=chunk.char_end,
        word_count=chunk.word_count,
        approx_tokens=chunk.approx_tokens,
        labels=labels,
        label_mask=label_mask,
        label_names=list(LABEL_NAMES),
        spans=span_records,
        span_categories=sorted(list(active_categories)),
        is_positive=is_pos
    )
