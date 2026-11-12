import { useQuery } from '@tanstack/react-query';
import { apiClient } from '../lib/api';

export function useBackendHealth() {
  const query = useQuery({
    queryKey: ['backend-health'],
    queryFn: () => apiClient.checkHealth(),
    refetchInterval: 30000,
    retry: 1,
    staleTime: 10000
  });

  const isMock =
    import.meta.env.VITE_USE_MOCK_API === 'true' ||
    import.meta.env.VITE_USE_MOCK_API === true;

  return {
    ...query,
    isMock,
    isHealthy: query.data?.status === 'healthy' || query.data?.status === 'ok',
    isOffline: query.isError || query.data?.status === 'offline'
  };
}

