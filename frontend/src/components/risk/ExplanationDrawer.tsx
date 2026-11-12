import React, { useEffect, useState } from 'react';
import { X, Copy, Check } from 'lucide-react';
import type { ExtractedClause, ReviewStatus } from '../../types/analysis';
import { RiskBadge } from '../ui/RiskBadge';
import { CategoryBadge } from '../ui/CategoryBadge';
import { ConfidenceBadge } from '../ui/ConfidenceBadge';
import { copyToClipboard } from '../../lib/utils';

interface ExplanationDrawerProps {
  clause: ExtractedClause | null;
  onClose: () => void;
  reviewStatus?: ReviewStatus;
  onStatusChange?: (clauseId: string, status: ReviewStatus) => void;
}

export const ExplanationDrawer: React.FC<ExplanationDrawerProps> = ({
  clause,
  onClose,
  reviewStatus = 'not_reviewed',
  onStatusChange
}) => {
  const [copiedClause, setCopiedClause] = useState(false);
  const [copiedMod, setCopiedMod] = useState(false);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onClose]);

  if (!clause) return null;

  const clauseKey = clause.id || `${clause.category}-${clause.start_char}`;

  const handleCopyClause = async () => {
    const ok = await copyToClipboard(clause.clause_text);
    if (ok) {
      setCopiedClause(true);
      setTimeout(() => setCopiedClause(false), 2000);
    }
  };

  const handleCopyMod = async () => {
    if (!clause.suggested_modification) return;
    const ok = await copyToClipboard(clause.suggested_modification);
    if (ok) {
      setCopiedMod(true);
      setTimeout(() => setCopiedMod(false), 2000);
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 flex justify-end bg-black/30 backdrop-blur-[1px] transition-opacity"
      role="dialog"
      aria-modal="true"
      aria-label="Clause details panel"
    >
      {/* Click outside to close */}
      <div className="flex-1" onClick={onClose} aria-hidden="true" />

      {/* Side Panel */}
      <div className="w-full max-w-lg bg-white dark:bg-[#181D26] h-full shadow-xl flex flex-col border-l border-[#E4E7EC] dark:border-[#28303E] overflow-hidden">
        {/* Panel Header */}
        <div className="p-4 border-b border-[#E4E7EC] dark:border-[#28303E] flex items-center justify-between gap-4">
          <div className="flex items-center gap-2 flex-wrap">
            <CategoryBadge category={clause.category} size="md" />
            <RiskBadge level={clause.risk_level} size="md" />
          </div>

          <button
            onClick={onClose}
            type="button"
            aria-label="Close clause details"
            className="p-1.5 rounded text-[#687385] hover:text-[#202938] dark:hover:text-white hover:bg-[#F1F3F6] dark:hover:bg-[#202633] transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Scrollable Content */}
        <div className="flex-1 overflow-y-auto custom-scrollbar p-5 space-y-6 text-sm">
          {/* Review Checklist Status Selector */}
          {onStatusChange && (
            <div className="p-3 bg-[#F7F8FA] dark:bg-[#1E242F] rounded border border-[#E4E7EC] dark:border-[#28303E] flex items-center justify-between gap-3 text-xs">
              <span className="font-medium text-[#202938] dark:text-[#F1F3F7]">
                Review status:
              </span>
              <div className="inline-flex rounded border border-[#D0D5DD] dark:border-[#384252] overflow-hidden bg-white dark:bg-[#11151C]">
                {(['not_reviewed', 'reviewed', 'needs_attention'] as ReviewStatus[]).map((status) => (
                  <button
                    key={status}
                    type="button"
                    onClick={() => onStatusChange(clauseKey, status)}
                    className={`px-2.5 py-1 text-[11px] font-medium transition-colors ${
                      reviewStatus === status
                        ? 'bg-[#315C8A] text-white'
                        : 'text-[#687385] dark:text-[#8B97A8] hover:bg-[#F1F3F6] dark:hover:bg-[#202633]'
                    }`}
                  >
                    {status === 'not_reviewed'
                      ? 'Not reviewed'
                      : status === 'reviewed'
                        ? 'Reviewed'
                        : 'Needs attention'}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* 1. What the contract says */}
          <section className="space-y-1.5">
            <div className="flex items-center justify-between">
              <h3 className="text-xs font-semibold text-[#687385] dark:text-[#8B97A8] uppercase tracking-wider">
                What the contract says
              </h3>
              <button
                onClick={handleCopyClause}
                type="button"
                className="inline-flex items-center gap-1 text-xs text-[#315C8A] dark:text-[#5D8EBF] hover:underline"
              >
                {copiedClause ? (
                  <>
                    <Check className="w-3.5 h-3.5 text-emerald-600" />
                    <span>Copied</span>
                  </>
                ) : (
                  <>
                    <Copy className="w-3.5 h-3.5" />
                    <span>Copy text</span>
                  </>
                )}
              </button>
            </div>
            <div className="p-3 bg-[#F7F8FA] dark:bg-[#11151C] border border-[#E4E7EC] dark:border-[#28303E] rounded font-serif text-xs text-[#202938] dark:text-[#F1F3F7] leading-relaxed whitespace-pre-wrap select-all">
              "{clause.clause_text}"
            </div>
          </section>

          {/* 2. What this means */}
          <section className="space-y-1.5">
            <h3 className="text-xs font-semibold text-[#687385] dark:text-[#8B97A8] uppercase tracking-wider">
              What this means
            </h3>
            <p className="text-[#202938] dark:text-[#F1F3F7] leading-relaxed">
              {clause.plain_english_explanation}
            </p>
          </section>

          {/* 3. Why it matters */}
          <section className="space-y-1.5">
            <h3 className="text-xs font-semibold text-[#687385] dark:text-[#8B97A8] uppercase tracking-wider">
              Why it matters
            </h3>
            <p className="text-[#202938] dark:text-[#F1F3F7] leading-relaxed">
              {clause.risk_reasoning}
            </p>
          </section>

          {/* 4. Risk assessment */}
          <section className="space-y-2">
            <h3 className="text-xs font-semibold text-[#687385] dark:text-[#8B97A8] uppercase tracking-wider">
              Risk assessment
            </h3>
            <div className="p-3 bg-[#F7F8FA] dark:bg-[#1E242F] border border-[#E4E7EC] dark:border-[#28303E] rounded space-y-1.5">
              <div className="flex items-center gap-2">
                <RiskBadge level={clause.risk_level} size="sm" />
                <ConfidenceBadge confidence={clause.confidence} size="sm" />
              </div>
              <p className="text-xs text-[#687385] dark:text-[#8B97A8] leading-relaxed">
                {clause.risk_level === 'CRITICAL' &&
                  'This clause significantly limits your rights or introduces serious financial liability. Negotiating this term is strongly recommended.'}
                {clause.risk_level === 'HIGH' &&
                  'This provision contains asymmetric obligations or restrictive conditions that warrant review before signing.'}
                {clause.risk_level === 'MEDIUM' &&
                  'This is a standard commercial condition that should be monitored during your engagement.'}
                {clause.risk_level === 'LOW' &&
                  'This represents balanced or customary contract language.'}
                {clause.risk_level === 'NEEDS_REVIEW' &&
                  'The clause references external terms or incomplete language. It could not be fully assessed automatically.'}
              </p>
            </div>
          </section>

          {/* 5. Suggested modification */}
          {clause.suggested_modification ? (
            <section className="space-y-1.5">
              <div className="flex items-center justify-between">
                <h3 className="text-xs font-semibold text-[#687385] dark:text-[#8B97A8] uppercase tracking-wider">
                  Suggested modification
                </h3>
                <button
                  onClick={handleCopyMod}
                  type="button"
                  className="inline-flex items-center gap-1 text-xs text-[#315C8A] dark:text-[#5D8EBF] hover:underline"
                >
                  {copiedMod ? (
                    <>
                      <Check className="w-3.5 h-3.5 text-emerald-600" />
                      <span>Copied</span>
                    </>
                  ) : (
                    <>
                      <Copy className="w-3.5 h-3.5" />
                      <span>Copy proposed text</span>
                    </>
                  )}
                </button>
              </div>
              <div className="p-3 bg-[#F7F8FA] dark:bg-[#11151C] border border-[#E4E7EC] dark:border-[#28303E] rounded text-xs text-[#202938] dark:text-[#F1F3F7] leading-relaxed whitespace-pre-wrap">
                {clause.suggested_modification}
              </div>
              <p className="text-[11px] text-[#687385] dark:text-[#8B97A8]">
                You can copy this text and propose it as redline language during contract review.
              </p>
            </section>
          ) : (
            <section className="space-y-1">
              <h3 className="text-xs font-semibold text-[#687385] dark:text-[#8B97A8] uppercase tracking-wider">
                Suggested modification
              </h3>
              <p className="text-xs text-[#687385] dark:text-[#8B97A8] italic">
                No specific counter-proposal is suggested for this customary provision.
              </p>
            </section>
          )}

          {/* Persistent Educational Disclaimer */}
          <div className="pt-4 border-t border-[#E4E7EC] dark:border-[#28303E] text-xs text-[#687385] dark:text-[#8B97A8] space-y-1">
            <p>
              <strong className="text-[#202938] dark:text-[#F1F3F7]">Educational notice:</strong>{' '}
              ClauseGuard is an educational document-analysis tool. It does not provide legal advice
              or replace consultation with a qualified legal professional.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
