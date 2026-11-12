import React from 'react';
import { Search, X } from 'lucide-react';
import type { ClauseCategory, RiskLevel } from '../../types/analysis';
import { ALL_CATEGORY_KEYS, getCategoryMeta } from '../../lib/categories';
import { RISK_ORDER, getRiskMeta } from '../../lib/risk';

interface RiskFiltersProps {
  searchQuery: string;
  onSearchChange: (q: string) => void;
  selectedCategory: ClauseCategory | 'ALL';
  onCategoryChange: (cat: ClauseCategory | 'ALL') => void;
  selectedRisk: RiskLevel | 'ALL';
  onRiskChange: (risk: RiskLevel | 'ALL') => void;
  onResetFilters: () => void;
  totalFilteredCount: number;
  totalCount: number;
}

export const RiskFilters: React.FC<RiskFiltersProps> = ({
  searchQuery,
  onSearchChange,
  selectedCategory,
  onCategoryChange,
  selectedRisk,
  onRiskChange,
  onResetFilters,
  totalFilteredCount,
  totalCount
}) => {
  const hasActiveFilters =
    searchQuery.trim().length > 0 ||
    selectedCategory !== 'ALL' ||
    selectedRisk !== 'ALL';

  return (
    <div className="p-3 bg-white dark:bg-[#181D26] border border-[#E4E7EC] dark:border-[#28303E] rounded space-y-2.5 text-xs">
      <div className="flex items-center justify-between">
        <span className="font-medium text-[#202938] dark:text-[#F1F3F7]">
          Filter clauses ({totalFilteredCount} of {totalCount})
        </span>

        {hasActiveFilters && (
          <button
            onClick={onResetFilters}
            type="button"
            className="inline-flex items-center gap-1 text-[#315C8A] dark:text-[#5D8EBF] hover:underline"
          >
            <X className="w-3 h-3" />
            <span>Reset filters</span>
          </button>
        )}
      </div>

      {/* Search Input */}
      <div className="relative">
        <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-[#687385]" />
        <input
          type="text"
          value={searchQuery}
          onChange={(e) => onSearchChange(e.target.value)}
          placeholder="Search clause text..."
          className="w-full pl-8 pr-2.5 py-1.5 rounded border border-[#D0D5DD] dark:border-[#384252] bg-white dark:bg-[#11151C] text-[#202938] dark:text-[#F1F3F7] placeholder:text-[#687385] focus:outline-none focus:border-[#315C8A]"
          aria-label="Filter clauses by text"
        />
      </div>

      {/* Dropdowns Row */}
      <div className="grid grid-cols-2 gap-2">
        {/* Category Dropdown */}
        <div>
          <label htmlFor="category-select" className="sr-only">
            Filter by Category
          </label>
          <select
            id="category-select"
            value={selectedCategory}
            onChange={(e) => onCategoryChange(e.target.value as ClauseCategory | 'ALL')}
            className="w-full py-1.5 px-2 rounded border border-[#D0D5DD] dark:border-[#384252] bg-white dark:bg-[#11151C] text-[#202938] dark:text-[#F1F3F7] focus:outline-none focus:border-[#315C8A]"
          >
            <option value="ALL">All categories</option>
            {ALL_CATEGORY_KEYS.map((key) => (
              <option key={key} value={key}>
                {getCategoryMeta(key).name}
              </option>
            ))}
          </select>
        </div>

        {/* Risk Level Dropdown */}
        <div>
          <label htmlFor="risk-select" className="sr-only">
            Filter by Risk Level
          </label>
          <select
            id="risk-select"
            value={selectedRisk}
            onChange={(e) => onRiskChange(e.target.value as RiskLevel | 'ALL')}
            className="w-full py-1.5 px-2 rounded border border-[#D0D5DD] dark:border-[#384252] bg-white dark:bg-[#11151C] text-[#202938] dark:text-[#F1F3F7] focus:outline-none focus:border-[#315C8A]"
          >
            <option value="ALL">All risk levels</option>
            {RISK_ORDER.map((level) => (
              <option key={level} value={level}>
                {getRiskMeta(level).label}
              </option>
            ))}
          </select>
        </div>
      </div>
    </div>
  );
};
