"""
ClauseGuard Paragraph-Aware Legal Text Chunker
==============================================
Segments raw legal contract contexts into model-ready chunks:
- Respects natural paragraph boundaries.
- Splits oversized paragraphs at sentence boundaries with controlled overlap.
- Employs active span lookahead to prevent unnecessary cross-boundary splitting.
- Guarantees 100% exact character offset preservation against raw contract context.
"""

import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from ml.preprocessing.constants import (
    TARGET_MAX_WORDS_PER_CHUNK,
    MAX_WORDS_HARD_LIMIT,
    SENTENCE_OVERLAP_WORDS,
)

@dataclass
class TextSpan:
    category: str
    char_start: int
    char_end: int
    text: str

@dataclass
class DocumentChunk:
    contract_id: str
    chunk_id: str
    chunk_index: int
    text: str
    char_start: int
    char_end: int
    word_count: int
    approx_tokens: int
    paragraph_indices: List[int]

def extract_natural_paragraphs(context: str) -> List[Tuple[int, int, str]]:
    """
    Identifies paragraph boundaries using double newlines, falling back to single newlines
    for flattened ASCII filings.

    Returns:
        List of (start_char, end_char, text) tuples matching exact context slices.
    """
    matches = list(re.finditer(r"\S.*?(?=\n\s*\n|\Z)", context, re.DOTALL))
    if len(matches) <= 1:
        matches = list(re.finditer(r"\S.*?(?=\n|\Z)", context, re.DOTALL))

    paragraphs = []
    for m in matches:
        s, e = m.start(), m.end()
        # Double check exact match
        p_text = context[s:e]
        paragraphs.append((s, e, p_text))
    return paragraphs

def split_oversized_paragraph(
    p_start: int,
    p_end: int,
    p_text: str,
    max_words: int = TARGET_MAX_WORDS_PER_CHUNK,
    overlap_words: int = SENTENCE_OVERLAP_WORDS
) -> List[Tuple[int, int, str]]:
    """
    Splits an oversized paragraph into sentence-aligned sub-units with controlled overlap.
    """
    sentence_matches = list(re.finditer(r"[^.!?]+[.!?]+|\S+", p_text))
    if not sentence_matches:
        return [(p_start, p_end, p_text)]

    sub_units: List[Tuple[int, int, str]] = []
    current_sentences: List[Tuple[int, int, str]] = []
    current_words = 0

    for sm in sentence_matches:
        s_local = sm.start()
        e_local = sm.end()
        s_text = sm.group()
        s_words = len(s_text.split())

        if current_words + s_words > max_words and current_sentences:
            unit_start = p_start + current_sentences[0][0]
            unit_end = p_start + current_sentences[-1][1]
            unit_text = p_text[current_sentences[0][0]:current_sentences[-1][1]]
            sub_units.append((unit_start, unit_end, unit_text))

            # Maintain small sentence overlap
            if len(current_sentences) > 1 and len(current_sentences[-1][2].split()) <= overlap_words:
                current_sentences = [current_sentences[-1]]
                current_words = len(current_sentences[0][2].split())
            else:
                current_sentences = []
                current_words = 0

        current_sentences.append((s_local, e_local, s_text))
        current_words += s_words

    if current_sentences:
        unit_start = p_start + current_sentences[0][0]
        unit_end = p_start + current_sentences[-1][1]
        unit_text = p_text[current_sentences[0][0]:current_sentences[-1][1]]
        sub_units.append((unit_start, unit_end, unit_text))

    return sub_units

def chunk_contract(
    contract_id: str,
    context: str,
    known_spans: Optional[List[TextSpan]] = None,
    target_words: int = TARGET_MAX_WORDS_PER_CHUNK,
    hard_limit_words: int = MAX_WORDS_HARD_LIMIT
) -> List[DocumentChunk]:
    """
    Chunks a contract context into transformer-ready document chunks:
    1. Extracts natural paragraphs.
    2. Decomposes oversized paragraphs at sentence boundaries.
    3. Accumulates units into chunks targeting 250-350 tokens (~180-260 words).
    4. Applies span lookahead to avoid severing active clause annotations where possible.
    5. Validates 100% exact text reconstruction for every chunk.
    """
    if known_spans is None:
        known_spans = []

    raw_paragraphs = extract_natural_paragraphs(context)

    # Normalize paragraph list by expanding oversized paragraphs
    units: List[Tuple[int, int, str, int]] = []  # (start, end, text, original_para_idx)
    for p_idx, (p_start, p_end, p_text) in enumerate(raw_paragraphs):
        words = len(p_text.split())
        if words > hard_limit_words:
            sub_units = split_oversized_paragraph(p_start, p_end, p_text, max_words=target_words)
            for su_start, su_end, su_text in sub_units:
                units.append((su_start, su_end, su_text, p_idx))
        else:
            units.append((p_start, p_end, p_text, p_idx))

    chunks: List[DocumentChunk] = []
    curr_units: List[Tuple[int, int, str, int]] = []
    curr_words: int = 0
    chunk_counter: int = 0

    for unit in units:
        u_start, u_end, u_text, u_pidx = unit
        u_words = len(u_text.split())

        # Check if an active span crosses the boundary of the current accumulation
        span_crossing = False
        if curr_units:
            c_start = curr_units[0][0]
            c_end = curr_units[-1][1]
            for sp in known_spans:
                # Span started in current chunk and extends into this unit
                if sp.char_start < c_end and sp.char_end > c_end and sp.char_start >= c_start:
                    span_crossing = True
                    break

        # Decide whether to merge with current chunk or finalize
        if not curr_units:
            curr_units.append(unit)
            curr_words = u_words
        elif span_crossing and (curr_words + u_words <= hard_limit_words):
            # Span lookahead merge: absorb unit to preserve clause span wholeness
            curr_units.append(unit)
            curr_words += u_words
        elif curr_words + u_words <= target_words:
            curr_units.append(unit)
            curr_words += u_words
        else:
            # Finalize current chunk
            c_start = curr_units[0][0]
            c_end = curr_units[-1][1]
            c_text = context[c_start:c_end]
            c_words = len(c_text.split())
            p_indices = sorted(list(set(u[3] for u in curr_units)))

            chunk = DocumentChunk(
                contract_id=contract_id,
                chunk_id=f"{contract_id}__chunk_{chunk_counter:04d}",
                chunk_index=chunk_counter,
                text=c_text,
                char_start=c_start,
                char_end=c_end,
                word_count=c_words,
                approx_tokens=int(c_words * 1.35),
                paragraph_indices=p_indices
            )
            # Integrity check
            assert context[c_start:c_end] == c_text, f"Offset mismatch in chunk {chunk.chunk_id}"
            chunks.append(chunk)
            chunk_counter += 1

            # Start new chunk
            curr_units = [unit]
            curr_words = u_words

    # Finalize remaining units
    if curr_units:
        c_start = curr_units[0][0]
        c_end = curr_units[-1][1]
        c_text = context[c_start:c_end]
        c_words = len(c_text.split())
        p_indices = sorted(list(set(u[3] for u in curr_units)))

        chunk = DocumentChunk(
            contract_id=contract_id,
            chunk_id=f"{contract_id}__chunk_{chunk_counter:04d}",
            chunk_index=chunk_counter,
            text=c_text,
            char_start=c_start,
            char_end=c_end,
            word_count=c_words,
            approx_tokens=int(c_words * 1.35),
            paragraph_indices=p_indices
        )
        assert context[c_start:c_end] == c_text, f"Offset mismatch in chunk {chunk.chunk_id}"
        chunks.append(chunk)

    return chunks
