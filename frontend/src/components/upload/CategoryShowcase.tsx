import React from 'react';
import { ALL_CATEGORY_KEYS, getCategoryMeta } from '../../lib/categories';

export const CategoryShowcase: React.FC = () => {
  return (
    <section className="space-y-3 pt-6 border-t border-[#E4E7EC] dark:border-[#28303E]">
      <div className="space-y-1">
        <h2 className="text-sm font-semibold text-[#202938] dark:text-[#F1F3F7]">
          What ClauseGuard checks
        </h2>
        <p className="text-xs text-[#687385] dark:text-[#8B97A8]">
          ClauseGuard scans agreements for nine standard contractual clause categories:
        </p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3 pt-1">
        {ALL_CATEGORY_KEYS.map((key) => {
          const meta = getCategoryMeta(key);
          return (
            <div
              key={key}
              className="p-3 rounded border border-[#E4E7EC] dark:border-[#28303E] bg-white dark:bg-[#181D26] space-y-1 text-left"
            >
              <h3 className="text-xs font-medium text-[#202938] dark:text-[#F1F3F7]">
                {meta.name}
              </h3>
              <p className="text-[11px] text-[#687385] dark:text-[#8B97A8] leading-relaxed">
                {meta.shortDescription}
              </p>
            </div>
          );
        })}
      </div>
    </section>
  );
};
