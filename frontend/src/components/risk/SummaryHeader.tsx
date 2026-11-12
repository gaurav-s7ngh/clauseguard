import React from 'react';
import { Download, CheckCircle2 } from 'lucide-react';
import type { AnalysisResult } from '../../types/analysis';
import { getRiskCounts } from '../../lib/risk';
import { exportAnalysisReportPDF } from '../../lib/export-pdf';

interface SummaryHeaderProps {
  result: AnalysisResult;
}

export const SummaryHeader: React.FC<SummaryHeaderProps> = ({ result }) => {
  const counts = getRiskCounts(result.clauses);

  const handleExport = () => {
    exportAnalysisReportPDF(result);
  };

  return (
    <div className="w-full bg-white dark:bg-[#181D26] border border-[#E4E7EC] dark:border-[#28303E] rounded p-4 transition-colors">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        {/* Document Title & Status */}
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <h1 className="text-base font-semibold text-[#202938] dark:text-[#F1F3F7]">
              {result.filename}
            </h1>
            <span className="inline-flex items-center gap-1 text-[11px] font-medium text-emerald-700 dark:text-emerald-400 bg-emerald-50 dark:bg-emerald-950/40 px-2 py-0.5 rounded border border-emerald-200 dark:border-emerald-900">
              <CheckCircle2 className="w-3 h-3" />
              <span>Analysis complete</span>
            </span>
          </div>
          <p className="text-xs text-[#687385] dark:text-[#8B97A8]">
            {result.total_extracted_clauses} total clauses detected
          </p>
        </div>

        {/* Compact Risk Breakdown & Download Button */}
        <div className="flex flex-wrap items-center gap-4 sm:gap-6">
          {/* Inline breakdown */}
          <div className="flex items-center gap-3 text-xs">
            {result.total_extracted_clauses === 0 && (
              <span className="text-emerald-700 dark:text-emerald-400 font-medium">
                No high-risk clauses detected
              </span>
            )}
            {counts.CRITICAL > 0 && (
              <span className="inline-flex items-center gap-1 text-red-700 dark:text-red-400 font-medium">
                <span className="w-2 h-2 rounded-full bg-red-600" />
                <span>{counts.CRITICAL} Critical</span>
              </span>
            )}
            {counts.HIGH > 0 && (
              <span className="inline-flex items-center gap-1 text-orange-700 dark:text-orange-400 font-medium">
                <span className="w-2 h-2 rounded-full bg-orange-500" />
                <span>{counts.HIGH} High</span>
              </span>
            )}
            {counts.MEDIUM > 0 && (
              <span className="inline-flex items-center gap-1 text-amber-700 dark:text-amber-400 font-medium">
                <span className="w-2 h-2 rounded-full bg-amber-500" />
                <span>{counts.MEDIUM} Medium</span>
              </span>
            )}
            {counts.LOW > 0 && (
              <span className="inline-flex items-center gap-1 text-emerald-700 dark:text-emerald-400 font-medium">
                <span className="w-2 h-2 rounded-full bg-emerald-600" />
                <span>{counts.LOW} Low</span>
              </span>
            )}
            {counts.NEEDS_REVIEW > 0 && (
              <span className="inline-flex items-center gap-1 text-slate-600 dark:text-slate-400 font-medium">
                <span className="w-2 h-2 rounded-full bg-slate-400" />
                <span>{counts.NEEDS_REVIEW} Needs Review</span>
              </span>
            )}
          </div>

          {/* Download Action */}
          <button
            onClick={handleExport}
            type="button"
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-medium bg-[#315C8A] hover:bg-[#264A70] text-white transition-colors"
          >
            <Download className="w-3.5 h-3.5" />
            <span>Download report</span>
          </button>
        </div>
      </div>
    </div>
  );
};
