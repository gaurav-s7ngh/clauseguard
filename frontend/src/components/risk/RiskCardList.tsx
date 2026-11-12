import React, { useMemo } from 'react';
import type { ExtractedClause, ReviewStatus } from '../../types/analysis';
import { sortClausesByRisk } from '../../lib/risk';
import { RiskCard } from './RiskCard';
import { FileQuestion, CheckCircle2 } from 'lucide-react';

interface RiskCardListProps {
  clauses: ExtractedClause[];
  selectedClause: ExtractedClause | null;
  onSelectClause: (clause: ExtractedClause) => void;
  reviewStatuses?: Record<string, ReviewStatus>;
  onStatusChange?: (clauseId: string, status: ReviewStatus) => void;
}

export const RiskCardList: React.FC<RiskCardListProps> = ({
  clauses,
  selectedClause,
  onSelectClause,
  reviewStatuses = {},
  onStatusChange
}) => {
  const sortedClauses = useMemo(() => {
    return sortClausesByRisk(clauses, 'desc');
  }, [clauses]);

  // Checklist counts
  const totalCount = clauses.length;
  const reviewedCount = clauses.filter((c) => {
    const key = c.id || `${c.category}-${c.start_char}`;
    return reviewStatuses[key] === 'reviewed';
  }).length;

  if (sortedClauses.length === 0) {
    return (
      <div className="p-8 text-center rounded border border-[#E4E7EC] dark:border-[#28303E] bg-white dark:bg-[#181D26] space-y-2">
        <FileQuestion className="w-6 h-6 mx-auto text-[#687385]" />
        <p className="text-xs font-medium text-[#202938] dark:text-[#F1F3F7]">
          No clauses match your filter
        </p>
        <p className="text-[11px] text-[#687385] dark:text-[#8B97A8]">
          Try selecting "All categories" or "All risk levels".
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {/* Secondary Checklist Progress */}
      {totalCount > 0 && onStatusChange && (
        <div className="px-3 py-2 bg-[#F1F3F6] dark:bg-[#202633] rounded flex items-center justify-between text-[11px] text-[#687385] dark:text-[#8B97A8]">
          <span className="flex items-center gap-1.5 font-medium text-[#202938] dark:text-[#F1F3F7]">
            <CheckCircle2 className="w-3.5 h-3.5 text-[#315C8A] dark:text-[#5D8EBF]" />
            <span>Review progress</span>
          </span>
          <span>
            {reviewedCount} of {totalCount} reviewed
          </span>
        </div>
      )}

      {/* Divided list */}
      <div className="border border-[#E4E7EC] dark:border-[#28303E] rounded bg-white dark:bg-[#181D26] divide-y divide-[#E4E7EC] dark:divide-[#28303E] overflow-hidden">
        {sortedClauses.map((clause) => {
          const isSelected = selectedClause
            ? clause.start_char === selectedClause.start_char &&
              clause.end_char === selectedClause.end_char &&
              clause.category === selectedClause.category
            : false;

          const clauseKey = clause.id || `${clause.category}-${clause.start_char}`;

          return (
            <RiskCard
              key={clauseKey}
              clause={clause}
              isSelected={isSelected}
              onSelect={onSelectClause}
              reviewStatus={reviewStatuses[clauseKey] || 'not_reviewed'}
              onStatusChange={onStatusChange}
            />
          );
        })}
      </div>
    </div>
  );
};
