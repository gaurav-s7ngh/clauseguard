import type { ClauseGuardApiClient } from './types';
import { RealClauseGuardApiClient } from './client';
import { MockClauseGuardApiClient } from './mock';

/**
 * Singleton API client instance.
 * Defaults to MockClauseGuardApiClient unless VITE_USE_MOCK_API is explicitly 'false'.
 */
function createApiClient(): ClauseGuardApiClient {
  const useMockEnv = import.meta.env.VITE_USE_MOCK_API;
  // Default to mock unless explicitly disabled:
  const useMock = useMockEnv === undefined || useMockEnv === 'true' || useMockEnv === true;

  if (useMock) {
    return new MockClauseGuardApiClient();
  }
  return new RealClauseGuardApiClient();
}

export const apiClient: ClauseGuardApiClient = createApiClient();
export * from './types';
export * from './adapters';
export * from './mock';
export * from './client';

