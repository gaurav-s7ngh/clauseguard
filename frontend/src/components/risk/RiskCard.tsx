import React from 'react';
import type { ExtractedClause, ReviewStatus } from '../../types/analysis';
import { RiskBadge } from '../ui/RiskBadge';
import { CategoryBadge } from '../ui/CategoryBadge';
import { ConfidenceBadge } from '../ui/ConfidenceBadge';
import { ChevronRight, Check, AlertCircle, CircleDot } from 'lucide-react';
import { cn } from '../../lib/utils';

interface RiskCardProps {
  clause: ExtractedClause;
  isSelected: boolean;
  onSelect: (clause: ExtractedClause) => void;
  reviewStatus?: ReviewStatus;
  onStatusChange?: (clauseId: string, status: ReviewStatus) => void;
}

export const RiskCard: React.FC<RiskCardProps> = ({
  clause,
  isSelected,
  onSelect,
  reviewStatus = 'not_reviewed',
  onStatusChange
}) => {
  const clauseKey = clause.id || `${clause.category}-${clause.start_char}`;

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      onSelect(clause);
    }
  };

  const handleCycleStatus = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (!onStatusChange) return;
    const nextStatus: ReviewStatus =
      reviewStatus === 'not_reviewed'
        ? 'reviewed'
        : reviewStatus === 'reviewed'
          ? 'needs_attention'
          : 'not_reviewed';
    onStatusChange(clauseKey, nextStatus);
  };

  return (
    <div
      onClick={() => onSelect(clause)}
      onKeyDown={handleKeyDown}
      tabIndex={0}
      role="button"
      aria-pressed={isSelected}
      aria-label={`Clause in ${clause.category.replace(/_/g, ' ')}. Risk: ${clause.risk_level}. Click to view details and scroll to text.`}
      className={cn(
        'p-3.5 transition-colors cursor-pointer text-left border-l-2 focus:outline-none focus-visible:bg-[#F1F3F6] dark:focus-visible:bg-[#202633]',
        isSelected
          ? 'bg-[#EDF3FA] dark:bg-[#1E2C3D] border-[#315C8A] dark:border-[#5D8EBF]'
          : 'bg-white dark:bg-[#181D26] border-transparent hover:bg-[#F7F8FA] dark:hover:bg-[#1E242F]'
      )}
    >
      {/* Top: Category, Risk Badge, and Checklist status */}
      <div className="flex items-center justify-between gap-2 mb-1.5">
        <div className="flex items-center gap-1.5 flex-wrap">
          <CategoryBadge category={clause.category} size="sm" />
          <RiskBadge level={clause.risk_level} size="sm" />
        </div>

        {/* Small Review Checklist Status Indicator */}
        {onStatusChange && (
          <button
            type="button"
            onClick={handleCycleStatus}
            title={`Checklist status: ${reviewStatus.replace('_', ' ')}. Click to toggle.`}
            className={cn(
              'inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-medium border transition-colors',
              reviewStatus === 'reviewed'
                ? 'bg-emerald-50 text-emerald-800 border-emerald-200 dark:bg-emerald-950/40 dark:text-emerald-300 dark:border-emerald-800'
                : reviewStatus === 'needs_attention'
                  ? 'bg-amber-50 text-amber-800 border-amber-200 dark:bg-amber-950/40 dark:text-amber-300 dark:border-amber-800'
                  : 'bg-[#F1F3F6] text-[#687385] border-[#E4E7EC] dark:bg-[#202633] dark:text-[#8B97A8] dark:border-[#28303E]'
            )}
          >
            {reviewStatus === 'reviewed' ? (
              <>
                <Check className="w-2.5 h-2.5 text-emerald-600" />
                <span>Reviewed</span>
              </>
            ) : reviewStatus === 'needs_attention' ? (
              <>
                <AlertCircle className="w-2.5 h-2.5 text-amber-600" />
                <span>Needs action</span>
              </>
            ) : (
              <>
                <CircleDot className="w-2.5 h-2.5 text-[#687385]" />
                <span>To review</span>
              </>
            )}
          </button>
        )}
      </div>

      {/* Short Plain-English Explanation */}
      <p className="text-xs text-[#202938] dark:text-[#F1F3F7] leading-relaxed line-clamp-2">
        {clause.plain_english_explanation}
      </p>

      {/* Footer: Detection Confidence & "View details" Action */}
      <div className="mt-2.5 pt-2 border-t border-[#E4E7EC] dark:border-[#28303E] flex items-center justify-between text-xs">
        <ConfidenceBadge confidence={clause.confidence} size="sm" />

        <span className="inline-flex items-center gap-0.5 font-medium text-[#315C8A] dark:text-[#5D8EBF] hover:underline">
          <span>View details</span>
          <ChevronRight className="w-3.5 h-3.5" />
        </span>
      </div>
    </div>
  );
};
