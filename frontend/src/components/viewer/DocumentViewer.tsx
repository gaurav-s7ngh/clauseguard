import React, { useMemo, useEffect, useRef, useState } from 'react';
import type { ExtractedClause } from '../../types/analysis';
import { segmentDocumentText } from '../../lib/highlight';
import { HighlightSegmentView } from './HighlightSegmentView';
import { ViewerToolbar } from './ViewerToolbar';
import { HighlightLegend } from './HighlightLegend';
import { cn } from '../../lib/utils';

interface DocumentViewerProps {
  documentText: string;
  clauses: ExtractedClause[];
  selectedClause: ExtractedClause | null;
  onSelectClause: (clause: ExtractedClause) => void;
  className?: string;
}

export const DocumentViewer: React.FC<DocumentViewerProps> = ({
  documentText,
  clauses,
  selectedClause,
  onSelectClause,
  className
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [fontSize, setFontSize] = useState<'sm' | 'base' | 'lg'>('base');

  // Compute character-accurate segments
  const segments = useMemo(() => {
    return segmentDocumentText(documentText, clauses);
  }, [documentText, clauses]);

  // Two-way sync: scroll to selected clause when selectedClause changes
  useEffect(() => {
    if (!selectedClause) return;

    const elId = `highlight-${selectedClause.start_char}-${selectedClause.end_char}`;
    const el = document.getElementById(elId);
    if (el) {
      el.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  }, [selectedClause]);

  // Ordered list of unique clauses by start position for prev/next navigation
  const sortedClauses = useMemo(() => {
    return [...clauses].sort((a, b) => a.start_char - b.start_char);
  }, [clauses]);

  const currentIndex = selectedClause
    ? sortedClauses.findIndex(
        (c) =>
          c.start_char === selectedClause.start_char &&
          c.end_char === selectedClause.end_char &&
          c.category === selectedClause.category
      )
    : -1;

  const handlePrevClause = () => {
    if (sortedClauses.length === 0) return;
    const nextIdx = currentIndex <= 0 ? sortedClauses.length - 1 : currentIndex - 1;
    onSelectClause(sortedClauses[nextIdx]);
  };

  const handleNextClause = () => {
    if (sortedClauses.length === 0) return;
    const nextIdx = currentIndex >= sortedClauses.length - 1 ? 0 : currentIndex + 1;
    onSelectClause(sortedClauses[nextIdx]);
  };

  const fontClasses = {
    sm: 'text-xs leading-relaxed',
    base: 'text-sm leading-relaxed',
    lg: 'text-base leading-loose'
  }[fontSize];

  return (
    <div
      className={cn(
        'flex flex-col h-full bg-white dark:bg-[#181D26] border border-[#E4E7EC] dark:border-[#28303E] rounded overflow-hidden shadow-xs transition-colors',
        className
      )}
    >
      {/* Header Toolbar */}
      <ViewerToolbar
        searchQuery={searchQuery}
        onSearchChange={setSearchQuery}
        currentClauseIndex={currentIndex}
        totalClauses={sortedClauses.length}
        onPrevClause={handlePrevClause}
        onNextClause={handleNextClause}
        fontSize={fontSize}
        onChangeFontSize={setFontSize}
        fullDocumentText={documentText}
      />

      {/* Main Document Content */}
      <div
        ref={containerRef}
        tabIndex={0}
        aria-label="Contract text viewer"
        className="flex-1 p-6 sm:p-8 overflow-y-auto custom-scrollbar font-serif text-[#202938] dark:text-[#F1F3F7] whitespace-pre-wrap selection:bg-[#D5E2F0] dark:selection:bg-[#2C4159] focus:outline-none"
      >
        <div className={cn('max-w-3xl mx-auto tracking-normal', fontClasses)}>
          {segments.map((segment) => (
            <HighlightSegmentView
              key={`seg-${segment.start_char}-${segment.end_char}`}
              segment={segment}
              selectedClause={selectedClause}
              onSelectClause={onSelectClause}
            />
          ))}
        </div>
      </div>

      {/* Bottom Legend */}
      <div className="p-3 border-t border-[#E4E7EC] dark:border-[#28303E] bg-[#F7F8FA] dark:bg-[#151921]">
        <HighlightLegend />
      </div>
    </div>
  );
};
