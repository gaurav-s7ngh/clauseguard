import React from 'react';
import { Loader2, AlertCircle, RefreshCw, CheckCircle2 } from 'lucide-react';
import type { DocumentStatus } from '../../types/analysis';

interface UploadProgressProps {
  status: DocumentStatus;
  progress: number;
  fileName?: string;
  errorMessage?: string | null;
  onRetry: () => void;
}

export const UploadProgress: React.FC<UploadProgressProps> = ({
  status,
  progress,
  fileName,
  errorMessage,
  onRetry
}) => {
  if (status === 'idle') return null;

  return (
    <div className="w-full p-4 bg-white dark:bg-[#181D26] rounded border border-[#E4E7EC] dark:border-[#28303E] space-y-3 transition-colors">
      <div className="flex items-center justify-between text-xs">
        <div className="flex items-center gap-2">
          {status === 'failed' ? (
            <AlertCircle className="w-4 h-4 text-red-600" />
          ) : status === 'completed' ? (
            <CheckCircle2 className="w-4 h-4 text-emerald-600" />
          ) : (
            <Loader2 className="w-4 h-4 text-[#315C8A] animate-spin" />
          )}

          <div>
            <span className="font-medium text-[#202938] dark:text-[#F1F3F7]">
              {status === 'uploading' && 'Uploading document...'}
              {status === 'analyzing' && 'Analyzing clauses and assessing risks...'}
              {status === 'completed' && 'Analysis ready.'}
              {status === 'failed' && 'Analysis failed.'}
            </span>
            {fileName && (
              <span className="text-[#687385] dark:text-[#8B97A8] ml-2">({fileName})</span>
            )}
          </div>
        </div>

        <span className="font-mono text-[#687385] dark:text-[#8B97A8]">{progress}%</span>
      </div>

      {status !== 'failed' && (
        <div className="h-1.5 w-full bg-[#F1F3F6] dark:bg-[#28303E] rounded-full overflow-hidden">
          <div
            className="h-full bg-[#315C8A] dark:bg-[#5D8EBF] rounded-full transition-all duration-300 ease-out"
            style={{ width: `${progress}%` }}
            role="progressbar"
            aria-valuenow={progress}
            aria-valuemin={0}
            aria-valuemax={100}
          />
        </div>
      )}

      {status === 'failed' && (
        <div className="flex items-center justify-between gap-4 pt-1 text-xs text-red-700 dark:text-red-300">
          <span>{errorMessage || 'An error occurred during analysis.'}</span>
          <button
            onClick={onRetry}
            type="button"
            className="inline-flex items-center gap-1 font-medium hover:underline text-[#315C8A] dark:text-[#5D8EBF]"
          >
            <RefreshCw className="w-3 h-3" />
            <span>Try again</span>
          </button>
        </div>
      )}
    </div>
  );
};
