import React from 'react';
import type { ClauseCategory } from '../../types/analysis';
import { getCategoryMeta } from '../../lib/categories';
import { cn } from '../../lib/utils';

interface CategoryBadgeProps {
  category: ClauseCategory;
  className?: string;
  size?: 'sm' | 'md';
}

export const CategoryBadge: React.FC<CategoryBadgeProps> = ({
  category,
  className,
  size = 'md'
}) => {
  const meta = getCategoryMeta(category);

  return (
    <span
      className={cn(
        'inline-flex items-center font-medium rounded text-[#315C8A] dark:text-[#5D8EBF] bg-[#EDF3FA] dark:bg-[#1E2C3D] border border-[#D5E2F0] dark:border-[#2C4159]',
        size === 'sm' ? 'text-[11px] px-2 py-0.5' : 'text-xs px-2.5 py-0.5',
        className
      )}
      title={meta.shortDescription}
    >
      <span>{meta.name}</span>
    </span>
  );
};
