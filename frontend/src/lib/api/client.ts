import type {
  AnalysisResult,
  AnalysisStatusResponse,
  DocumentHistoryItem,
  HealthCheckResponse,
  UploadResponse
} from '../../types/analysis';
import type { ClauseGuardApiClient, RawAnalysisResultDTO } from './types';
import { adaptAnalysisResult } from './adapters';
import { getMockAnalysisResult } from './mock';

export class RealClauseGuardApiClient implements ClauseGuardApiClient {
  private baseUrl: string;

  constructor(baseUrl?: string) {
    this.baseUrl = (
      baseUrl ||
      (import.meta.env.VITE_API_BASE_URL as string) ||
      'http://127.0.0.1:8000'
    ).replace(/\/+$/, '');
  }

  private async request<T>(endpoint: string, options?: RequestInit): Promise<T> {
    const url = `${this.baseUrl}${endpoint.startsWith('/') ? endpoint : `/${endpoint}`}`;
    try {
      const response = await fetch(url, {
        ...options,
        headers: {
          Accept: 'application/json',
          ...(options?.headers || {})
        }
      });

      if (!response.ok) {
        let errorMessage = `HTTP ${response.status}: ${response.statusText}`;
        try {
          const errorJson = await response.json();
          if (errorJson.detail) {
            errorMessage = typeof errorJson.detail === 'string'
              ? errorJson.detail
              : JSON.stringify(errorJson.detail);
          } else if (errorJson.message) {
            errorMessage = errorJson.message;
          }
        } catch {
          // ignore non-json error body
        }
        throw new Error(errorMessage);
      }

      return (await response.json()) as T;
    } catch (err: unknown) {
      if (err instanceof Error) {
        if (err.message.includes('Failed to fetch') || err.message.includes('NetworkError')) {
          throw new Error('ClauseGuard backend is offline. Please ensure the backend is running at ' + this.baseUrl, { cause: err });
        }
        throw err;
      }
      throw new Error(`Network request to ${url} failed`, { cause: err });
    }
  }

  async uploadDocument(file: File): Promise<UploadResponse> {
    const formData = new FormData();
    formData.append('file', file);

    const url = `${this.baseUrl}/api/v1/documents/upload`;
    try {
      const response = await fetch(url, {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        let msg = `Upload failed with status ${response.status}`;
        try {
          const err = await response.json();
          msg = err.detail || err.message || msg;
        } catch {
          // use default msg
        }
        throw new Error(msg);
      }

      return await response.json();
    } catch (err: unknown) {
      if (err instanceof Error) {
        if (err.message.includes('Failed to fetch') || err.message.includes('NetworkError')) {
          throw new Error('ClauseGuard backend is offline. Please ensure the backend is running at ' + this.baseUrl, { cause: err });
        }
        throw err;
      }
      throw new Error('Document upload failed', { cause: err });
    }
  }

  async startAnalysis(documentId: string): Promise<AnalysisStatusResponse> {
    if (documentId.startsWith('doc_mock_') || documentId.startsWith('doc_179')) {
      return {
        document_id: documentId,
        status: 'completed',
        progress: 100,
        message: 'Sample analysis completed.'
      };
    }
    return this.request<AnalysisStatusResponse>(`/api/v1/analyze/${documentId}`, {
      method: 'POST'
    });
  }

  async getAnalysisStatus(documentId: string): Promise<AnalysisStatusResponse> {
    if (documentId.startsWith('doc_mock_') || documentId.startsWith('doc_179')) {
      return {
        document_id: documentId,
        status: 'completed',
        progress: 100,
        message: 'Sample analysis completed.'
      };
    }
    try {
      return await this.request<AnalysisStatusResponse>(`/api/v1/status/${documentId}`);
    } catch {
      // If no dedicated /status endpoint, fallback to querying /results
      try {
        const results = await this.getResults(documentId);
        if (results && results.clauses) {
          return { document_id: documentId, status: 'completed', progress: 100 };
        }
      } catch {
        // still processing or not found
      }
      return { document_id: documentId, status: 'processing', progress: 50 };
    }
  }

  async getResults(documentId: string): Promise<AnalysisResult> {
    if (documentId.startsWith('doc_mock_') || documentId.startsWith('doc_179')) {
      return getMockAnalysisResult(documentId);
    }
    try {
      const raw = await this.request<RawAnalysisResultDTO>(`/api/v1/results/${documentId}`);
      return adaptAnalysisResult(raw);
    } catch (err: unknown) {
      if (documentId.startsWith('doc_mock_') || documentId.startsWith('doc_179') || documentId.includes('sample')) {
        return getMockAnalysisResult(documentId);
      }
      throw err;
    }
  }

  async checkHealth(): Promise<HealthCheckResponse> {
    return this.request<HealthCheckResponse>('/api/v1/health');
  }

  async getDocumentHistory(): Promise<DocumentHistoryItem[]> {
    try {
      return await this.request<DocumentHistoryItem[]>('/api/v1/documents');
    } catch {
      return [];
    }
  }
}
