import React from 'react';
import { AlertOctagon, AlertTriangle, Info, CheckCircle, HelpCircle } from 'lucide-react';

export const HighlightLegend: React.FC = () => {
  return (
    <div className="flex flex-wrap items-center gap-3 text-xs text-[#687385] dark:text-[#8B97A8]">
      <span className="font-medium text-[#202938] dark:text-[#F1F3F7]">
        Highlights:
      </span>

      <span className="inline-flex items-center gap-1 text-[11px] text-red-800 dark:text-red-300">
        <AlertOctagon className="w-3 h-3 text-red-600 dark:text-red-400" />
        Critical
      </span>

      <span className="inline-flex items-center gap-1 text-[11px] text-orange-800 dark:text-orange-300">
        <AlertTriangle className="w-3 h-3 text-orange-600 dark:text-orange-400" />
        High
      </span>

      <span className="inline-flex items-center gap-1 text-[11px] text-amber-800 dark:text-amber-300">
        <Info className="w-3 h-3 text-amber-600 dark:text-amber-400" />
        Medium
      </span>

      <span className="inline-flex items-center gap-1 text-[11px] text-emerald-800 dark:text-emerald-300">
        <CheckCircle className="w-3 h-3 text-emerald-600 dark:text-emerald-400" />
        Low
      </span>

      <span className="inline-flex items-center gap-1 text-[11px] text-slate-700 dark:text-slate-300">
        <HelpCircle className="w-3 h-3 text-slate-500 dark:text-slate-400" />
        Needs Review
      </span>
    </div>
  );
};
