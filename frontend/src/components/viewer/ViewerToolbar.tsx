import React from 'react';
import {
  Search,
  ChevronUp,
  ChevronDown,
  Copy,
  Check
} from 'lucide-react';
import { copyToClipboard } from '../../lib/utils';

interface ViewerToolbarProps {
  searchQuery: string;
  onSearchChange: (q: string) => void;
  currentClauseIndex: number;
  totalClauses: number;
  onPrevClause: () => void;
  onNextClause: () => void;
  fontSize: 'sm' | 'base' | 'lg';
  onChangeFontSize: (size: 'sm' | 'base' | 'lg') => void;
  fullDocumentText: string;
}

export const ViewerToolbar: React.FC<ViewerToolbarProps> = ({
  searchQuery,
  onSearchChange,
  currentClauseIndex,
  totalClauses,
  onPrevClause,
  onNextClause,
  fontSize,
  onChangeFontSize,
  fullDocumentText
}) => {
  const [copied, setCopied] = React.useState(false);

  const handleCopyText = async () => {
    const success = await copyToClipboard(fullDocumentText);
    if (success) {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  return (
    <div className="flex flex-wrap items-center justify-between gap-3 p-3 bg-white dark:bg-[#181D26] border-b border-[#E4E7EC] dark:border-[#28303E]">
      {/* Search in document */}
      <div className="relative flex-1 min-w-[180px] max-w-xs">
        <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-[#687385]" />
        <input
          type="text"
          value={searchQuery}
          onChange={(e) => onSearchChange(e.target.value)}
          placeholder="Search in document..."
          className="w-full pl-8 pr-2.5 py-1.5 text-xs rounded border border-[#D0D5DD] dark:border-[#384252] bg-white dark:bg-[#11151C] text-[#202938] dark:text-[#F1F3F7] placeholder:text-[#687385] focus:outline-none focus:border-[#315C8A]"
          aria-label="Search contract text"
        />
      </div>

      <div className="flex items-center gap-2">
        {/* Clause Navigator */}
        {totalClauses > 0 && (
          <div className="flex items-center gap-1 text-xs text-[#687385] dark:text-[#8B97A8]">
            <span className="font-mono">
              {currentClauseIndex >= 0 ? `${currentClauseIndex + 1} / ${totalClauses}` : `${totalClauses} clauses`}
            </span>
            <button
              onClick={onPrevClause}
              type="button"
              title="Previous clause"
              aria-label="Jump to previous clause"
              className="p-1 rounded hover:bg-[#F1F3F6] dark:hover:bg-[#202633] text-[#202938] dark:text-[#F1F3F7]"
            >
              <ChevronUp className="w-3.5 h-3.5" />
            </button>
            <button
              onClick={onNextClause}
              type="button"
              title="Next clause"
              aria-label="Jump to next clause"
              className="p-1 rounded hover:bg-[#F1F3F6] dark:hover:bg-[#202633] text-[#202938] dark:text-[#F1F3F7]"
            >
              <ChevronDown className="w-3.5 h-3.5" />
            </button>
          </div>
        )}

        <div className="h-3.5 w-px bg-[#E4E7EC] dark:bg-[#28303E]" />

        {/* Font Size Adjuster */}
        <div className="flex items-center rounded border border-[#D0D5DD] dark:border-[#384252] overflow-hidden text-xs">
          <button
            onClick={() =>
              onChangeFontSize(fontSize === 'lg' ? 'base' : fontSize === 'base' ? 'sm' : 'sm')
            }
            type="button"
            title="Smaller font"
            aria-label="Decrease text size"
            className="px-2 py-1 hover:bg-[#F1F3F6] dark:hover:bg-[#202633] text-[#687385] dark:text-[#8B97A8]"
          >
            A-
          </button>
          <button
            onClick={() =>
              onChangeFontSize(fontSize === 'sm' ? 'base' : fontSize === 'base' ? 'lg' : 'lg')
            }
            type="button"
            title="Larger font"
            aria-label="Increase text size"
            className="px-2 py-1 hover:bg-[#F1F3F6] dark:hover:bg-[#202633] text-[#202938] dark:text-[#F1F3F7] font-medium"
          >
            A+
          </button>
        </div>

        {/* Copy full document */}
        <button
          onClick={handleCopyText}
          type="button"
          title="Copy full document text"
          aria-label="Copy full contract text"
          className="inline-flex items-center gap-1 px-2.5 py-1 text-xs rounded border border-[#D0D5DD] dark:border-[#384252] bg-white dark:bg-[#181D26] hover:bg-[#F1F3F6] dark:hover:bg-[#202633] text-[#202938] dark:text-[#F1F3F7] transition-colors"
        >
          {copied ? (
            <>
              <Check className="w-3 h-3 text-emerald-600" />
              <span>Copied</span>
            </>
          ) : (
            <>
              <Copy className="w-3 h-3 text-[#687385]" />
              <span>Copy</span>
            </>
          )}
        </button>
      </div>
    </div>
  );
};
