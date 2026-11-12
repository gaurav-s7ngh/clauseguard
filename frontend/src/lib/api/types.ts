import type {
  AnalysisResult,
  UploadResponse,
  AnalysisStatusResponse,
  HealthCheckResponse,
  DocumentHistoryItem
} from '../../types/analysis';

export interface ClauseGuardApiClient {
  /**
   * Upload a contract document (PDF, DOCX, TXT)
   */
  uploadDocument(file: File): Promise<UploadResponse>;

  /**
   * Trigger analysis for an uploaded document
   */
  startAnalysis(documentId: string): Promise<AnalysisStatusResponse>;

  /**
   * Poll or check current analysis status
   */
  getAnalysisStatus(documentId: string): Promise<AnalysisStatusResponse>;

  /**
   * Fetch complete structured analysis results
   */
  getResults(documentId: string): Promise<AnalysisResult>;

  /**
   * Backend health check readiness probe
   */
  checkHealth(): Promise<HealthCheckResponse>;

  /**
   * Get document history (stubbed for mock / Stage 6 future endpoint)
   */
  getDocumentHistory(): Promise<DocumentHistoryItem[]>;
}

/**
 * Raw backend DTO representations (Stage 5/6 may vary slightly)
 * Handled via src/lib/api/adapters.ts
 */
export interface RawClauseDTO {
  category: string;
  clause_text: string;
  start_char: number;
  end_char: number;
  confidence: number;
  chunk_id?: string;
  document_id?: string;
  page_number?: number | null;
  surrounding_context?: string;
  model_version?: string;
  extraction_status?: string;
  // Possible Stage 5 field name variations:
  risk_level?: string;
  severity?: string;
  risk_score?: number;
  score?: number;
  risk_reasoning?: string;
  reasoning?: string;
  plain_english_explanation?: string;
  explanation?: string;
  suggested_modification?: string;
  recommendation?: string;
}

export interface RawAnalysisResultDTO {
  document_id: string;
  filename?: string;
  document_name?: string;
  document_text: string;
  document_char_length?: number;
  total_chunks?: number;
  total_extracted_clauses?: number;
  pipeline_version?: string;
  clauses?: RawClauseDTO[];
}
