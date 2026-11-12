import { describe, it, expect } from 'vitest';
import { segmentDocumentText, isSegmentInClause } from '../lib/highlight';
import type { ExtractedClause, ClauseCategory, RiskLevel } from '../types/analysis';
import { SAMPLE_CONTRACT_TEXT, getMockSampleClauses } from '../lib/api/mock';

function createDummyClause(
  category: ClauseCategory,
  start: number,
  end: number,
  risk: RiskLevel = 'HIGH',
  text = ''
): ExtractedClause {
  return {
    id: `clause-${start}-${end}`,
    category,
    start_char: start,
    end_char: end,
    clause_text: text,
    confidence: 0.9,
    chunk_id: 'c1',
    document_id: 'd1',
    page_number: 1,
    surrounding_context: text,
    model_version: 'v1',
    extraction_status: 'extracted',
    risk_level: risk,
    risk_reasoning: 'Test reason',
    plain_english_explanation: 'Test expl'
  };
}

describe('Highlighting Engine (segmentDocumentText)', () => {
  it('handles empty document text gracefully', () => {
    const segments = segmentDocumentText('');
    expect(segments).toEqual([]);
  });

  it('returns single unhighlighted segment when no clauses are provided', () => {
    const doc = 'This is an ordinary contract with no clauses.';
    const segments = segmentDocumentText(doc, []);
    expect(segments.length).toBe(1);
    expect(segments[0].is_highlighted).toBe(false);
    expect(segments[0].start_char).toBe(0);
    expect(segments[0].end_char).toBe(doc.length);
    expect(segments[0].text).toBe(doc);
  });

  it('handles non-overlapping disjoint clauses accurately', () => {
    const doc = '0123456789ABCDEFGHIJ'; // 20 chars
    const clauses = [
      createDummyClause('liability_cap', 2, 5, 'HIGH', '234'),
      createDummyClause('audit_rights', 10, 15, 'LOW', 'ABCDE')
    ];

    const segments = segmentDocumentText(doc, clauses);
    // Expected segments: [0,2), [2,5), [5,10), [10,15), [15,20)
    expect(segments.length).toBe(5);

    expect(segments[0].text).toBe('01');
    expect(segments[0].is_highlighted).toBe(false);

    expect(segments[1].text).toBe('234');
    expect(segments[1].is_highlighted).toBe(true);
    expect(segments[1].primary_risk).toBe('HIGH');
    expect(segments[1].clauses.length).toBe(1);

    expect(segments[2].text).toBe('56789');
    expect(segments[2].is_highlighted).toBe(false);

    expect(segments[3].text).toBe('ABCDE');
    expect(segments[3].is_highlighted).toBe(true);
    expect(segments[3].primary_risk).toBe('LOW');

    expect(segments[4].text).toBe('FGHIJ');
    expect(segments[4].is_highlighted).toBe(false);

    // Concatenation check
    expect(segments.map((s) => s.text).join('')).toBe(doc);
  });

  it('handles overlapping clauses without offset drift', () => {
    const doc = 'ABCDEFGHIJKLM'; // 13 chars
    // Clause A covers 2 to 7 ('CDEFG')
    // Clause B covers 5 to 10 ('FGHIJ')
    const clauses = [
      createDummyClause('liability_cap', 2, 7, 'CRITICAL'),
      createDummyClause('audit_rights', 5, 10, 'MEDIUM')
    ];

    const segments = segmentDocumentText(doc, clauses);
    // Boundaries: 0, 2, 5, 7, 10, 13
    expect(segments.length).toBe(5);

    // [0, 2) 'AB'
    expect(segments[0].text).toBe('AB');
    expect(segments[0].is_highlighted).toBe(false);

    // [2, 5) 'CDE' (Clause A only)
    expect(segments[1].text).toBe('CDE');
    expect(segments[1].is_highlighted).toBe(true);
    expect(segments[1].clauses.length).toBe(1);
    expect(segments[1].primary_risk).toBe('CRITICAL');

    // [5, 7) 'FG' (Both Clause A and B overlap!)
    expect(segments[2].text).toBe('FG');
    expect(segments[2].is_highlighted).toBe(true);
    expect(segments[2].clauses.length).toBe(2);
    // Highest severity: CRITICAL vs MEDIUM is CRITICAL
    expect(segments[2].primary_risk).toBe('CRITICAL');

    // [7, 10) 'HIJ' (Clause B only)
    expect(segments[3].text).toBe('HIJ');
    expect(segments[3].is_highlighted).toBe(true);
    expect(segments[3].clauses.length).toBe(1);
    expect(segments[3].primary_risk).toBe('MEDIUM');

    // [10, 13) 'KLM'
    expect(segments[4].text).toBe('KLM');
    expect(segments[4].is_highlighted).toBe(false);

    expect(segments.map((s) => s.text).join('')).toBe(doc);
  });

  it('handles nested clauses accurately', () => {
    const doc = 'START_OUTER_INNER_OUTER_END';
    // Outer covers [6, 23): 'OUTER_INNER_OUTER'
    // Inner covers [12, 17): 'INNER'
    const clauses = [
      createDummyClause('non_compete', 6, 23, 'HIGH'),
      createDummyClause('anti_assignment', 12, 17, 'CRITICAL')
    ];

    const segments = segmentDocumentText(doc, clauses);
    // Boundaries: 0, 6, 12, 17, 23, doc.length
    expect(segments.length).toBe(5);

    expect(segments[0].text).toBe('START_');
    expect(segments[1].text).toBe('OUTER_');
    expect(segments[1].primary_risk).toBe('HIGH');

    expect(segments[2].text).toBe('INNER');
    expect(segments[2].clauses.length).toBe(2);
    // Inner is CRITICAL, Outer is HIGH => primary is CRITICAL
    expect(segments[2].primary_risk).toBe('CRITICAL');

    expect(segments[3].text).toBe('_OUTER');
    expect(segments[3].primary_risk).toBe('HIGH');

    expect(segments[4].text).toBe('_END');

    expect(segments.map((s) => s.text).join('')).toBe(doc);
  });

  it('handles adjacent clauses with zero-gap boundary', () => {
    const doc = 'ABCDEFGHIJ';
    const clauses = [
      createDummyClause('license_grant', 0, 5, 'LOW'),
      createDummyClause('governing_law', 5, 10, 'LOW')
    ];

    const segments = segmentDocumentText(doc, clauses);
    expect(segments.length).toBe(2);
    expect(segments[0].text).toBe('ABCDE');
    expect(segments[1].text).toBe('FGHIJ');
    expect(segments.map((s) => s.text).join('')).toBe(doc);
  });

  it('handles exact boundary offsets (start at 0, end at document length)', () => {
    const doc = 'FULL_SPAN_DOCUMENT';
    const clauses = [createDummyClause('liability_cap', 0, doc.length, 'CRITICAL')];

    const segments = segmentDocumentText(doc, clauses);
    expect(segments.length).toBe(1);
    expect(segments[0].start_char).toBe(0);
    expect(segments[0].end_char).toBe(doc.length);
    expect(segments[0].text).toBe(doc);
    expect(segments[0].is_highlighted).toBe(true);
    expect(segments[0].primary_risk).toBe('CRITICAL');
  });

  it('PROVES string concatenation equals original text on complex sample contract', () => {
    const clauses = getMockSampleClauses();
    const segments = segmentDocumentText(SAMPLE_CONTRACT_TEXT, clauses);

    // Strict equality check on full 14k characters
    const concatenated = segments.map((s) => s.text).join('');
    expect(concatenated).toBe(SAMPLE_CONTRACT_TEXT);
    expect(concatenated.length).toBe(SAMPLE_CONTRACT_TEXT.length);
  });

  it('correctly tests isSegmentInClause helper', () => {
    const doc = 'ABCDEFGHIJ';
    const targetClause = createDummyClause('governing_law', 2, 7, 'LOW');
    const otherClause = createDummyClause('license_grant', 8, 10, 'LOW');

    const segments = segmentDocumentText(doc, [targetClause, otherClause]);
    // segment for [2, 7)
    const targetSegment = segments.find((s) => s.start_char === 2 && s.end_char === 7);
    expect(targetSegment).toBeDefined();
    expect(isSegmentInClause(targetSegment!, targetClause)).toBe(true);
    expect(isSegmentInClause(targetSegment!, otherClause)).toBe(false);
    expect(isSegmentInClause(targetSegment!, null)).toBe(false);
  });
});
