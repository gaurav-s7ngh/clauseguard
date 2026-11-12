import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { DocumentViewer } from '../components/viewer/DocumentViewer';
import { RiskCardList } from '../components/risk/RiskCardList';
import type { ExtractedClause } from '../types/analysis';

const mockDocumentText =
  'Preamble -- Liability capped at $100. Middle -- Renews automatically for 24 months. End.';

const clause1Text = 'Liability capped at $100.';
const start1 = mockDocumentText.indexOf(clause1Text);
const end1 = start1 + clause1Text.length;

const clause2Text = 'Renews automatically for 24 months.';
const start2 = mockDocumentText.indexOf(clause2Text);
const end2 = start2 + clause2Text.length;

const mockClauses: ExtractedClause[] = [
  {
    id: 'clause-1',
    category: 'liability_cap',
    clause_text: clause1Text,
    start_char: start1,
    end_char: end1,
    confidence: 0.95,
    chunk_id: 'chunk_0',
    document_id: 'doc_1',
    page_number: 1,
    surrounding_context: 'Context around liability',
    model_version: 'v1',
    extraction_status: 'extracted',
    risk_level: 'CRITICAL',
    risk_score: 9,
    risk_reasoning: 'Nominal liability disclaimer.',
    plain_english_explanation: 'Maximum payout is only $100.',
    suggested_modification: 'Cap at 12 months fees.'
  },
  {
    id: 'clause-2',
    category: 'renewal_term',
    clause_text: clause2Text,
    start_char: start2,
    end_char: end2,
    confidence: 0.92,
    chunk_id: 'chunk_0',
    document_id: 'doc_1',
    page_number: 1,
    surrounding_context: 'Context around renewal',
    model_version: 'v1',
    extraction_status: 'extracted',
    risk_level: 'HIGH',
    risk_score: 8,
    risk_reasoning: 'Unreasonable 24-month renewal.',
    plain_english_explanation: 'Locks you into two more years.',
    suggested_modification: 'Renews for 12 months with 30-day notice.'
  }
];

describe('Two-Way Synchronization: RiskCard <-> DocumentViewer', () => {
  beforeEach(() => {
    // Mock scrollIntoView in jsdom
    window.HTMLElement.prototype.scrollIntoView = vi.fn();
  });

  it('triggers onSelect when user clicks a RiskCard', () => {
    const handleSelect = vi.fn();
    render(
      <RiskCardList
        clauses={mockClauses}
        selectedClause={null}
        onSelectClause={handleSelect}
      />
    );

    const card = screen.getByRole('button', { name: /liability cap/i });
    expect(card).toBeInTheDocument();

    fireEvent.click(card);

    expect(handleSelect).toHaveBeenCalledTimes(1);
    expect(handleSelect).toHaveBeenCalledWith(mockClauses[0]);
  });

  it('triggers onSelectClause when user clicks a highlighted mark in DocumentViewer', () => {
    const handleSelect = vi.fn();
    render(
      <DocumentViewer
        documentText={mockDocumentText}
        clauses={mockClauses}
        selectedClause={null}
        onSelectClause={handleSelect}
      />
    );

    // Look for the highlighted mark
    const mark = screen.getByText(clause1Text);
    expect(mark.tagName).toBe('MARK');

    fireEvent.click(mark);

    expect(handleSelect).toHaveBeenCalledTimes(1);
    expect(handleSelect).toHaveBeenCalledWith(
      expect.objectContaining({
        category: 'liability_cap',
        start_char: start1,
        end_char: end1
      })
    );
  });

  it('applies active highlight styles and calls scrollIntoView when selectedClause is set', () => {
    const selected = mockClauses[0];
    render(
      <DocumentViewer
        documentText={mockDocumentText}
        clauses={mockClauses}
        selectedClause={selected}
        onSelectClause={vi.fn()}
      />
    );

    const mark = screen.getByText(clause1Text);
    expect(mark).toHaveClass('ring-2 ring-indigo-500');
    expect(window.HTMLElement.prototype.scrollIntoView).toHaveBeenCalled();
  });

  it('reflects aria-pressed="true" on the corresponding RiskCard when selected', () => {
    const selected = mockClauses[1];
    render(
      <RiskCardList
        clauses={mockClauses}
        selectedClause={selected}
        onSelectClause={vi.fn()}
      />
    );

    const renewalCard = screen.getByRole('button', { name: /renewal term/i });
    expect(renewalCard).toHaveAttribute('aria-pressed', 'true');

    const liabilityCard = screen.getByRole('button', { name: /liability cap/i });
    expect(liabilityCard).toHaveAttribute('aria-pressed', 'false');
  });
});
