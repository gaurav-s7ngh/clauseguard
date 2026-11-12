/**
 * ClauseGuard Analysis Types & Schemas
 * Authoritative types matching ML Pipeline (Stages 3-5) and Frontend Display (Stage 7)
 */

export type ClauseCategory =
  | 'liability_cap'
  | 'termination_for_convenience'
  | 'anti_assignment'
  | 'renewal_term'
  | 'governing_law'
  | 'audit_rights'
  | 'ip_ownership_assignment'
  | 'license_grant'
  | 'non_compete';

export type RiskLevel = 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW' | 'NEEDS_REVIEW';

export type ReviewStatus = 'not_reviewed' | 'reviewed' | 'needs_attention';

export interface ExtractedClause {
  category: ClauseCategory;
  clause_text: string;
  start_char: number; // global offset into original document text
  end_char: number; // exclusive
  confidence: number; // detection confidence 0-1 (algorithm certainty, NOT risk severity)
  chunk_id: string;
  document_id: string;
  page_number: number | null;
  surrounding_context: string;
  model_version: string;
  extraction_status: string;
  // Stage 5 risk fields:
  risk_level: RiskLevel;
  risk_score?: number; // 1 to 10 or undefined
  risk_reasoning: string;
  plain_english_explanation: string;
  suggested_modification?: string;
  // UI identifier helper (computed if needed)
  id?: string;
}

export interface AnalysisResult {
  document_id: string;
  filename: string;
  document_text: string; // full original text, never modified
  document_char_length: number;
  total_chunks: number;
  total_extracted_clauses: number;
  pipeline_version: string;
  clauses: ExtractedClause[];
}

export type DocumentStatus = 'idle' | 'uploading' | 'analyzing' | 'completed' | 'failed';

export interface UploadResponse {
  document_id: string;
  filename: string;
  status: string;
  message?: string;
}

export interface AnalysisStatusResponse {
  document_id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress?: number;
  message?: string;
}

export interface HealthCheckResponse {
  status: 'healthy' | 'ok' | 'degraded' | 'offline';
  version?: string;
  model_loaded?: boolean;
}

export interface DocumentHistoryItem {
  document_id: string;
  filename: string;
  uploaded_at: string;
  total_clauses: number;
  critical_count: number;
  high_count: number;
  medium_count: number;
  low_count: number;
  needs_review_count: number;
}

/**
 * Highlighting interval partition segment
 */
export interface HighlightSegment {
  start_char: number;
  end_char: number;
  text: string;
  clauses: ExtractedClause[]; // empty if background normal text
  is_highlighted: boolean;
  primary_risk?: RiskLevel; // highest risk among active clauses
}

