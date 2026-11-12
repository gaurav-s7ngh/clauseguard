import type {
  AnalysisResult,
  ClauseCategory,
  ExtractedClause,
  RiskLevel
} from '../../types/analysis';
import type { RawAnalysisResultDTO, RawClauseDTO } from './types';

const VALID_CATEGORIES: Set<ClauseCategory> = new Set([
  'liability_cap',
  'termination_for_convenience',
  'anti_assignment',
  'renewal_term',
  'governing_law',
  'audit_rights',
  'ip_ownership_assignment',
  'license_grant',
  'non_compete'
]);

const VALID_RISK_LEVELS: Set<RiskLevel> = new Set([
  'CRITICAL',
  'HIGH',
  'MEDIUM',
  'LOW',
  'NEEDS_REVIEW'
]);

/**
 * Normalizes risk level strings into canonical RiskLevel enum
 */
export function normalizeRiskLevel(raw?: string): RiskLevel {
  if (!raw) return 'NEEDS_REVIEW';
  const upper = raw.trim().toUpperCase();
  if (VALID_RISK_LEVELS.has(upper as RiskLevel)) {
    return upper as RiskLevel;
  }
  if (upper === 'CRIT' || upper === 'SEVERE') return 'CRITICAL';
  if (upper === 'MODERATE' || upper === 'MED') return 'MEDIUM';
  if (upper === 'INFO' || upper === 'SAFE') return 'LOW';
  return 'NEEDS_REVIEW';
}

/**
 * Normalizes category strings into canonical ClauseCategory
 */
export function normalizeCategory(raw: string): ClauseCategory {
  const clean = raw.trim().toLowerCase().replace(/[-\s]/g, '_');
  if (VALID_CATEGORIES.has(clean as ClauseCategory)) {
    return clean as ClauseCategory;
  }
  // Fallbacks for known CUAD aliases if passed raw
  if (clean.includes('cap') || clean.includes('liability')) return 'liability_cap';
  if (clean.includes('termination')) return 'termination_for_convenience';
  if (clean.includes('assignment')) return 'anti_assignment';
  if (clean.includes('renewal')) return 'renewal_term';
  if (clean.includes('governing') || clean.includes('law')) return 'governing_law';
  if (clean.includes('audit')) return 'audit_rights';
  if (clean.includes('ip') || clean.includes('ownership')) return 'ip_ownership_assignment';
  if (clean.includes('license')) return 'license_grant';
  if (clean.includes('compete')) return 'non_compete';

  return 'liability_cap'; // safe fallback
}

/**
 * Adapts a raw clause DTO from backend/ML pipeline into strongly-typed ExtractedClause
 * Isolate all field renaming here.
 */
export function adaptRawClause(raw: RawClauseDTO, index: number): ExtractedClause {
  const category = normalizeCategory(raw.category);
  const risk_level = normalizeRiskLevel(raw.risk_level || raw.severity);

  // Fallback defaults if Stage 5 has missing or renamed text fields
  const risk_reasoning =
    raw.risk_reasoning ||
    raw.reasoning ||
    (risk_level === 'NEEDS_REVIEW'
      ? 'The automated model could not conclusively determine the risk profile of this clause.'
      : `Flagged under standard legal risk heuristics for ${category.replace(/_/g, ' ')}.`);

  const plain_english_explanation =
    raw.plain_english_explanation ||
    raw.explanation ||
    'This clause specifies contractual terms that may limit your rights or impose ongoing obligations. Review the exact wording carefully.';

  const suggested_modification =
    raw.suggested_modification ||
    raw.recommendation ||
    (risk_level === 'HIGH' || risk_level === 'CRITICAL'
      ? 'Consider negotiating mutual obligations or adding reasonable notice and liability carve-outs.'
      : undefined);

  return {
    id: `clause-${index}-${raw.start_char}-${raw.end_char}`,
    category,
    clause_text: raw.clause_text || '',
    start_char: Number(raw.start_char),
    end_char: Number(raw.end_char),
    confidence: typeof raw.confidence === 'number' ? raw.confidence : 0.85,
    chunk_id: raw.chunk_id || `chunk_${index}`,
    document_id: raw.document_id || 'doc_unknown',
    page_number: raw.page_number ?? null,
    surrounding_context: raw.surrounding_context || raw.clause_text || '',
    model_version: raw.model_version || 'roberta-span-v1',
    extraction_status: raw.extraction_status || 'extracted',
    risk_level,
    risk_score: raw.risk_score ?? raw.score,
    risk_reasoning,
    plain_english_explanation,
    suggested_modification
  };
}

/**
 * Adapts raw analysis result from backend into UI AnalysisResult
 */
export function adaptAnalysisResult(raw: RawAnalysisResultDTO): AnalysisResult {
  const rawClauses = Array.isArray(raw.clauses) ? raw.clauses : [];
  const adaptedClauses = rawClauses.map((c, i) => adaptRawClause(c, i));

  return {
    document_id: raw.document_id,
    filename: raw.filename || raw.document_name || 'contract.txt',
    document_text: raw.document_text || '',
    document_char_length: raw.document_char_length || raw.document_text?.length || 0,
    total_chunks: raw.total_chunks || Math.ceil((raw.document_text?.length || 0) / 1000),
    total_extracted_clauses: adaptedClauses.length,
    pipeline_version: raw.pipeline_version || 'clauseguard-v1.0.0',
    clauses: adaptedClauses
  };
}

