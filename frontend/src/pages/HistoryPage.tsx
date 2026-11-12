import React from 'react';
import { Link } from 'react-router-dom';
import { Plus } from 'lucide-react';
import { useDocumentHistory } from '../hooks/useAnalysis';
import { HistoryTable } from '../components/history/HistoryTable';

export const HistoryPage: React.FC = () => {
  const { data: documents, isLoading } = useDocumentHistory();

  return (
    <div className="flex-1 py-8 px-4 sm:px-6 max-w-5xl mx-auto w-full space-y-6">
      <div className="flex items-center justify-between border-b border-[#E4E7EC] dark:border-[#28303E] pb-4">
        <div>
          <h1 className="text-xl font-semibold text-[#202938] dark:text-[#F1F3F7]">
            Document history
          </h1>
          <p className="text-xs text-[#687385] dark:text-[#8B97A8] mt-0.5">
            Previously analyzed contracts and clause assessments.
          </p>
        </div>

        <Link
          to="/"
          className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-medium bg-[#315C8A] hover:bg-[#264A70] text-white transition-colors"
        >
          <Plus className="w-3.5 h-3.5" />
          <span>Upload contract</span>
        </Link>
      </div>

      <HistoryTable documents={documents || []} isLoading={isLoading} />
    </div>
  );
};
