import { useQuery } from '@tanstack/react-query';
import { apiClient } from '../lib/api';
import type { AnalysisResult } from '../types/analysis';

export function useAnalysis(documentId?: string) {
  return useQuery<AnalysisResult, Error>({
    queryKey: ['document-analysis', documentId],
    queryFn: async () => {
      if (!documentId) throw new Error('No document ID provided');
      return await apiClient.getResults(documentId);
    },
    enabled: Boolean(documentId),
    refetchInterval: (query) => {
      // If completed or errored, stop polling
      const data = query.state.data;
      if (data && data.clauses && data.clauses.length > 0) {
        return false;
      }
      // Poll every 2 seconds during active processing
      return 2000;
    },
    retry: 2,
    staleTime: 60000
  });
}

export function useDocumentHistory() {
  return useQuery({
    queryKey: ['document-history'],
    queryFn: () => apiClient.getDocumentHistory(),
    staleTime: 15000
  });
}

