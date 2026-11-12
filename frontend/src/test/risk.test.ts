import { describe, it, expect } from 'vitest';
import {
  sortClausesByRisk,
  getHighestRisk,
  getRiskCounts,
  getRiskMeta
} from '../lib/risk';
import type { ExtractedClause, RiskLevel, ClauseCategory } from '../types/analysis';

function makeClause(
  category: ClauseCategory,
  risk_level: RiskLevel,
  start_char: number,
  clause_text: string = ''
): ExtractedClause {
  return {
    id: `c-${category}-${start_char}`,
    category,
    clause_text,
    start_char,
    end_char: start_char + 50,
    confidence: 0.9,
    chunk_id: 'c1',
    document_id: 'd1',
    page_number: 1,
    surrounding_context: clause_text,
    model_version: 'v1',
    extraction_status: 'extracted',
    risk_level,
    risk_reasoning: 'Reason',
    plain_english_explanation: 'Expl'
  };
}

describe('Risk Utilities & Sorting (risk.ts)', () => {
  it('defines correct rank hierarchy for all risk levels', () => {
    expect(getRiskMeta('CRITICAL').rank).toBeGreaterThan(getRiskMeta('HIGH').rank);
    expect(getRiskMeta('HIGH').rank).toBeGreaterThan(getRiskMeta('MEDIUM').rank);
    expect(getRiskMeta('MEDIUM').rank).toBeGreaterThan(getRiskMeta('LOW').rank);
    expect(getRiskMeta('LOW').rank).toBeGreaterThan(getRiskMeta('NEEDS_REVIEW').rank);
  });

  it('sorts clauses in strict severity order (CRITICAL to LOW, then NEEDS_REVIEW)', () => {
    const clauses: ExtractedClause[] = [
      makeClause('governing_law', 'LOW', 100),
      makeClause('audit_rights', 'NEEDS_REVIEW', 200),
      makeClause('liability_cap', 'CRITICAL', 50),
      makeClause('renewal_term', 'HIGH', 300),
      makeClause('license_grant', 'MEDIUM', 150)
    ];

    const sorted = sortClausesByRisk(clauses, 'desc');
    expect(sorted.map((c) => c.risk_level)).toEqual([
      'CRITICAL',
      'HIGH',
      'MEDIUM',
      'LOW',
      'NEEDS_REVIEW'
    ]);
  });

  it('breaks ties using start_char position ascending', () => {
    const clauses: ExtractedClause[] = [
      makeClause('liability_cap', 'CRITICAL', 500),
      makeClause('ip_ownership_assignment', 'CRITICAL', 100),
      makeClause('non_compete', 'CRITICAL', 250)
    ];

    const sorted = sortClausesByRisk(clauses, 'desc');
    expect(sorted.map((c) => c.start_char)).toEqual([100, 250, 500]);
  });

  it('calculates highest risk correctly among combinations', () => {
    expect(getHighestRisk([])).toBeNull();
    expect(getHighestRisk(['LOW', 'MEDIUM', 'HIGH'])).toBe('HIGH');
    expect(getHighestRisk(['LOW', 'CRITICAL', 'HIGH'])).toBe('CRITICAL');
    expect(getHighestRisk(['NEEDS_REVIEW', 'LOW'])).toBe('LOW');
    expect(getHighestRisk(['NEEDS_REVIEW'])).toBe('NEEDS_REVIEW');
  });

  it('accurately tallies risk counts across a collection', () => {
    const clauses: ExtractedClause[] = [
      makeClause('liability_cap', 'CRITICAL', 10),
      makeClause('ip_ownership_assignment', 'CRITICAL', 20),
      makeClause('renewal_term', 'HIGH', 30),
      makeClause('license_grant', 'MEDIUM', 40),
      makeClause('governing_law', 'LOW', 50),
      makeClause('audit_rights', 'NEEDS_REVIEW', 60)
    ];

    const counts = getRiskCounts(clauses);
    expect(counts).toEqual({
      CRITICAL: 2,
      HIGH: 1,
      MEDIUM: 1,
      LOW: 1,
      NEEDS_REVIEW: 1
    });
  });

  it('filters clauses across category, risk level, and text query', () => {
    const clauses: ExtractedClause[] = [
      makeClause('liability_cap', 'CRITICAL', 10, 'Limitation of liability aggregate $100'),
      makeClause('renewal_term', 'HIGH', 20, 'Agreement automatically renews for 24 months'),
      makeClause('governing_law', 'LOW', 30, 'Delaware courts jurisdiction'),
      makeClause('audit_rights', 'NEEDS_REVIEW', 40, 'Exhibit D inspection')
    ];

    // Filter by risk
    const criticalOnly = clauses.filter((c) => c.risk_level === 'CRITICAL');
    expect(criticalOnly.length).toBe(1);
    expect(criticalOnly[0].category).toBe('liability_cap');

    // Filter by category
    const renewalOnly = clauses.filter((c) => c.category === 'renewal_term');
    expect(renewalOnly.length).toBe(1);

    // Text search (case-insensitive)
    const query = 'delaware';
    const textMatches = clauses.filter((c) =>
      c.clause_text.toLowerCase().includes(query.toLowerCase())
    );
    expect(textMatches.length).toBe(1);
    expect(textMatches[0].category).toBe('governing_law');
  });
});
