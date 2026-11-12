import React from 'react';
import type { ExtractedClause, HighlightSegment } from '../../types/analysis';
import { getRiskMeta } from '../../lib/risk';
import { getCategoryMeta } from '../../lib/categories';
import { cn } from '../../lib/utils';

interface HighlightSegmentViewProps {
  segment: HighlightSegment;
  selectedClause: ExtractedClause | null;
  onSelectClause: (clause: ExtractedClause) => void;
}

export const HighlightSegmentView: React.FC<HighlightSegmentViewProps> = React.memo(
  ({ segment, selectedClause, onSelectClause }) => {
    if (!segment.is_highlighted || segment.clauses.length === 0) {
      return <span>{segment.text}</span>;
    }

    const primaryRisk = segment.primary_risk || 'NEEDS_REVIEW';
    const riskMeta = getRiskMeta(primaryRisk);

    const isSelected = selectedClause
      ? segment.clauses.some(
          (c) =>
            c.start_char === selectedClause.start_char &&
            c.end_char === selectedClause.end_char &&
            c.category === selectedClause.category
        )
      : false;

    // Active clause to trigger on click
    const activeClause =
      segment.clauses.find(
        (c) =>
          selectedClause &&
          c.start_char === selectedClause.start_char &&
          c.end_char === selectedClause.end_char
      ) || segment.clauses[0];

    const categoryNames = segment.clauses
      .map((c) => getCategoryMeta(c.category).name)
      .join(', ');

    const handleKeyDown = (e: React.KeyboardEvent) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        onSelectClause(activeClause);
      }
    };

    return (
      <mark
        id={`highlight-${activeClause.start_char}-${activeClause.end_char}`}
        onClick={() => onSelectClause(activeClause)}
        onKeyDown={handleKeyDown}
        tabIndex={0}
        role="button"
        aria-label={`Clause: ${categoryNames}. Risk level: ${riskMeta.label}. Click to inspect.`}
        title={`${categoryNames} (${riskMeta.label}) - Click to inspect`}
        className={cn(
          'cursor-pointer transition-colors rounded-xs px-0.5 inline',
          riskMeta.highlightBgLight,
          riskMeta.highlightBorderLight,
          riskMeta.highlightBgDark,
          riskMeta.highlightBorderDark,
          isSelected &&
            'ring-2 ring-indigo-500 ring-offset-1 bg-indigo-200/80 dark:bg-indigo-900/70 font-medium'
        )}
      >
        {segment.text}
      </mark>
    );
  }
);

HighlightSegmentView.displayName = 'HighlightSegmentView';
