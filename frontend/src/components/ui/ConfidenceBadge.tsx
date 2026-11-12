import React from 'react';
import { formatConfidence, cn } from '../../lib/utils';

interface ConfidenceBadgeProps {
  confidence: number;
  className?: string;
  size?: 'sm' | 'md';
}

export const ConfidenceBadge: React.FC<ConfidenceBadgeProps> = ({
  confidence,
  className,
  size = 'md'
}) => {
  const formatted = formatConfidence(confidence);

  return (
    <span
      className={cn(
        'inline-flex items-center gap-1 text-[#687385] dark:text-[#8B97A8]',
        size === 'sm' ? 'text-xs' : 'text-xs',
        className
      )}
      title="Detection confidence: Algorithmic certainty in detecting and categorizing this clause. Does not represent risk level."
      aria-label={`Detection confidence: ${formatted}`}
    >
      <span>Detection confidence:</span>
      <span className="font-medium text-[#202938] dark:text-[#F1F3F7]">{formatted}</span>
    </span>
  );
};
