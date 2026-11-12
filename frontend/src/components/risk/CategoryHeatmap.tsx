import React from 'react';
import type { ClauseCategory, ExtractedClause, RiskLevel } from '../../types/analysis';
import { ALL_CATEGORY_KEYS, getCategoryMeta } from '../../lib/categories';
import { getHighestRisk, getRiskMeta } from '../../lib/risk';
import { cn } from '../../lib/utils';
import { MinusCircle } from 'lucide-react';

interface CategoryHeatmapProps {
  clauses: ExtractedClause[];
  selectedCategory: ClauseCategory | null;
  onSelectCategory: (category: ClauseCategory | null) => void;
}

export const CategoryHeatmap: React.FC<CategoryHeatmapProps> = ({
  clauses,
  selectedCategory,
  onSelectCategory
}) => {
  // Group clauses by category
  const categoryMap = new Map<ClauseCategory, ExtractedClause[]>();
  for (const cat of ALL_CATEGORY_KEYS) {
    categoryMap.set(cat, []);
  }
  for (const c of clauses) {
    categoryMap.get(c.category)?.push(c);
  }

  return (
    <div className="w-full bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-4 shadow-xs space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="text-xs font-bold uppercase tracking-wider text-slate-700 dark:text-slate-300">
          9-Category Risk Heatmap
        </h3>
        {selectedCategory && (
          <button
            onClick={() => onSelectCategory(null)}
            type="button"
            className="text-xs text-indigo-600 dark:text-indigo-400 hover:underline font-medium"
          >
            Clear Filter
          </button>
        )}
      </div>

      <div className="grid grid-cols-3 sm:grid-cols-9 gap-2">
        {ALL_CATEGORY_KEYS.map((catKey) => {
          const catMeta = getCategoryMeta(catKey);
          const foundClauses = categoryMap.get(catKey) || [];
          const hasDetected = foundClauses.length > 0;
          const highestRisk: RiskLevel | null = hasDetected
            ? getHighestRisk(foundClauses.map((c) => c.risk_level))
            : null;

          const riskMeta = highestRisk ? getRiskMeta(highestRisk) : null;
          const isFilterActive = selectedCategory === catKey;

          return (
            <button
              key={catKey}
              onClick={() => onSelectCategory(isFilterActive ? null : catKey)}
              type="button"
              title={`${catMeta.name}: ${
                hasDetected ? `${highestRisk} (${foundClauses.length} clause${foundClauses.length > 1 ? 's' : ''})` : 'Not detected'
              }`}
              aria-label={`Filter by ${catMeta.name}. Status: ${
                hasDetected ? highestRisk : 'Not detected'
              }`}
              className={cn(
                'flex flex-col items-center justify-between p-2 rounded-xl border text-center transition-all focus:outline-none focus:ring-2 focus:ring-indigo-500',
                hasDetected && riskMeta
                  ? `${riskMeta.badgeBgLight} ${riskMeta.badgeBorderLight} ${riskMeta.badgeBgDark} ${riskMeta.badgeBorderDark} hover:scale-105`
                  : 'bg-slate-50 dark:bg-slate-800/40 border-slate-200 dark:border-slate-800 text-slate-400 opacity-60 hover:opacity-90',
                isFilterActive && 'ring-2 ring-indigo-600 dark:ring-indigo-400 scale-105 shadow-sm'
              )}
            >
              <div className="w-full text-[10px] font-bold truncate text-slate-800 dark:text-slate-200">
                {catMeta.name}
              </div>

              <div className="my-1">
                {hasDetected && riskMeta ? (
                  <span
                    className={cn(
                      'text-[9px] font-extrabold uppercase px-1.5 py-0.5 rounded-full inline-block',
                      riskMeta.badgeTextLight,
                      riskMeta.badgeTextDark
                    )}
                  >
                    {riskMeta.shortLabel}
                  </span>
                ) : (
                  <span className="text-[9px] text-slate-400 inline-flex items-center gap-0.5">
                    <MinusCircle className="w-2.5 h-2.5" />
                    None
                  </span>
                )}
              </div>

              <span className="text-[9px] text-slate-500 dark:text-slate-400 font-mono">
                {foundClauses.length} found
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
};
