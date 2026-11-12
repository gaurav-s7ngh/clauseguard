import type { ExtractedClause, HighlightSegment, RiskLevel } from '../types/analysis';
import { getHighestRisk } from './risk';

/**
 * Segment the raw document text into contiguous, non-overlapping intervals
 * defined by the union of all clause boundaries.
 *
 * Guarantees:
 * 1. Absolute character preservation:
 *    segments.map(s => s.text).join('') === documentText
 * 2. Exact start_char and end_char offsets matching the original string.
 * 3. Handles disjoint, adjacent, nested, and overlapping clause spans without drift.
 * 4. Each segment tracks all active overlapping clauses and computes the primary risk.
 */
export function segmentDocumentText(
  documentText: string,
  clauses: ExtractedClause[] = []
): HighlightSegment[] {
  const docLen = documentText.length;
  if (docLen === 0) {
    return [];
  }

  // Filter and sanitize clause bounds
  const validClauses = clauses.filter((c) => {
    return (
      typeof c.start_char === 'number' &&
      typeof c.end_char === 'number' &&
      c.start_char >= 0 &&
      c.end_char <= docLen &&
      c.start_char < c.end_char
    );
  });

  if (validClauses.length === 0) {
    return [
      {
        start_char: 0,
        end_char: docLen,
        text: documentText,
        clauses: [],
        is_highlighted: false,
        primary_risk: undefined
      }
    ];
  }

  // Collect and sort all unique boundary points
  const boundaryPoints = new Set<number>([0, docLen]);
  for (const clause of validClauses) {
    boundaryPoints.add(clause.start_char);
    boundaryPoints.add(clause.end_char);
  }

  const sortedPoints = Array.from(boundaryPoints).sort((a, b) => a - b);
  const segments: HighlightSegment[] = [];

  for (let i = 0; i < sortedPoints.length - 1; i++) {
    const start = sortedPoints[i];
    const end = sortedPoints[i + 1];

    if (start === end) continue;

    // Find all clauses covering the interval [start, end)
    const coveringClauses = validClauses.filter(
      (c) => c.start_char <= start && c.end_char >= end
    );

    const isHighlighted = coveringClauses.length > 0;
    const primaryRisk: RiskLevel | undefined = isHighlighted
      ? getHighestRisk(coveringClauses.map((c) => c.risk_level)) || undefined
      : undefined;

    segments.push({
      start_char: start,
      end_char: end,
      text: documentText.slice(start, end),
      clauses: coveringClauses,
      is_highlighted: isHighlighted,
      primary_risk: primaryRisk
    });
  }

  return segments;
}

/**
 * Helper to determine if a segment belongs to a specific selected clause
 */
export function isSegmentInClause(
  segment: HighlightSegment,
  selectedClause: ExtractedClause | null
): boolean {
  if (!selectedClause) return false;
  return segment.clauses.some(
    (c) =>
      c.start_char === selectedClause.start_char &&
      c.end_char === selectedClause.end_char &&
      c.category === selectedClause.category
  );
}

