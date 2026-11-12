import React from 'react';
import { Link } from 'react-router-dom';
import { FileText, ChevronRight } from 'lucide-react';
import type { DocumentHistoryItem } from '../../types/analysis';

interface HistoryTableProps {
  documents: DocumentHistoryItem[];
  isLoading?: boolean;
}

export const HistoryTable: React.FC<HistoryTableProps> = ({ documents, isLoading }) => {
  if (isLoading) {
    return (
      <div className="w-full p-8 text-center text-xs text-[#687385]">
        Loading history...
      </div>
    );
  }

  if (documents.length === 0) {
    return (
      <div className="p-8 text-center rounded border border-[#E4E7EC] dark:border-[#28303E] bg-white dark:bg-[#181D26] space-y-3">
        <FileText className="w-8 h-8 mx-auto text-[#687385]" />
        <div className="space-y-1">
          <p className="text-sm font-medium text-[#202938] dark:text-[#F1F3F7]">
            No analyzed contracts yet
          </p>
          <p className="text-xs text-[#687385] dark:text-[#8B97A8]">
            Upload a contract to identify clauses, analyze risks and generate explanations.
          </p>
        </div>
        <Link
          to="/"
          className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-medium bg-[#315C8A] hover:bg-[#264A70] text-white transition-colors"
        >
          <span>Upload contract</span>
        </Link>
      </div>
    );
  }

  return (
    <div className="w-full bg-white dark:bg-[#181D26] border border-[#E4E7EC] dark:border-[#28303E] rounded overflow-hidden">
      <table className="w-full text-left text-xs">
        <thead className="bg-[#F7F8FA] dark:bg-[#151921] border-b border-[#E4E7EC] dark:border-[#28303E] text-[#687385] dark:text-[#8B97A8] font-medium">
          <tr>
            <th className="py-3 px-4">Document</th>
            <th className="py-3 px-4">Date</th>
            <th className="py-3 px-4">Clauses</th>
            <th className="py-3 px-4">Risk Breakdown</th>
            <th className="py-3 px-4 text-right">Action</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-[#E4E7EC] dark:divide-[#28303E]">
          {documents.map((doc) => (
            <tr
              key={doc.document_id}
              className="hover:bg-[#F7F8FA] dark:hover:bg-[#1E242F] transition-colors"
            >
              <td className="py-3.5 px-4">
                <span className="font-medium text-[#202938] dark:text-[#F1F3F7] block">
                  {doc.filename}
                </span>
                <span className="text-[11px] font-mono text-[#687385] dark:text-[#8B97A8]">
                  {doc.document_id}
                </span>
              </td>

              <td className="py-3.5 px-4 text-[#687385] dark:text-[#8B97A8]">
                {new Date(doc.uploaded_at).toLocaleDateString()}
              </td>

              <td className="py-3.5 px-4 text-[#202938] dark:text-[#F1F3F7]">
                {doc.total_clauses}
              </td>

              <td className="py-3.5 px-4">
                <div className="flex items-center gap-2">
                  {doc.critical_count > 0 && (
                    <span className="text-red-700 dark:text-red-400 font-medium">
                      {doc.critical_count} Critical
                    </span>
                  )}
                  {doc.high_count > 0 && (
                    <span className="text-orange-700 dark:text-orange-400 font-medium">
                      {doc.high_count} High
                    </span>
                  )}
                  {doc.critical_count === 0 && doc.high_count === 0 && (
                    <span className="text-emerald-700 dark:text-emerald-400">
                      Standard terms
                    </span>
                  )}
                </div>
              </td>

              <td className="py-3.5 px-4 text-right">
                <Link
                  to={`/documents/${doc.document_id}`}
                  className="inline-flex items-center gap-0.5 text-[#315C8A] dark:text-[#5D8EBF] hover:underline font-medium"
                >
                  <span>Open</span>
                  <ChevronRight className="w-3.5 h-3.5" />
                </Link>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};
