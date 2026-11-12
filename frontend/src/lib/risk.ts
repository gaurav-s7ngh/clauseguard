import type { ExtractedClause, RiskLevel } from '../types/analysis';

export interface RiskMetadata {
  level: RiskLevel;
  label: string;
  shortLabel: string;
  rank: number; // 5 = CRITICAL, 4 = HIGH, 3 = MEDIUM, 2 = LOW, 1 = NEEDS_REVIEW
  description: string;
  badgeBgLight: string;
  badgeTextLight: string;
  badgeBorderLight: string;
  badgeBgDark: string;
  badgeTextDark: string;
  badgeBorderDark: string;
  highlightBgLight: string;
  highlightBorderLight: string;
  highlightBgDark: string;
  highlightBorderDark: string;
  iconName: string;
}

export const RISK_LEVELS: Record<RiskLevel, RiskMetadata> = {
  CRITICAL: {
    level: 'CRITICAL',
    label: 'Critical Risk',
    shortLabel: 'Critical',
    rank: 5,
    description: 'Substantial legal liability or one-sided restriction requiring immediate negotiation.',
    badgeBgLight: 'bg-red-50',
    badgeTextLight: 'text-red-800',
    badgeBorderLight: 'border-red-200',
    badgeBgDark: 'dark:bg-red-950/40',
    badgeTextDark: 'dark:text-red-300',
    badgeBorderDark: 'dark:border-red-900',
    highlightBgLight: 'bg-red-100/70 hover:bg-red-200/80',
    highlightBorderLight: 'border-b border-red-500',
    highlightBgDark: 'dark:bg-red-950/60 dark:hover:bg-red-900/70',
    highlightBorderDark: 'dark:border-red-400',
    iconName: 'AlertOctagon'
  },
  HIGH: {
    level: 'HIGH',
    label: 'High Risk',
    shortLabel: 'High',
    rank: 4,
    description: 'Burdensome condition or asymmetry that warrants caution and legal review.',
    badgeBgLight: 'bg-orange-50',
    badgeTextLight: 'text-orange-800',
    badgeBorderLight: 'border-orange-200',
    badgeBgDark: 'dark:bg-orange-950/40',
    badgeTextDark: 'dark:text-orange-300',
    badgeBorderDark: 'dark:border-orange-900',
    highlightBgLight: 'bg-orange-100/70 hover:bg-orange-200/80',
    highlightBorderLight: 'border-b border-orange-500',
    highlightBgDark: 'dark:bg-orange-950/60 dark:hover:bg-orange-900/70',
    highlightBorderDark: 'dark:border-orange-400',
    iconName: 'AlertTriangle'
  },
  MEDIUM: {
    level: 'MEDIUM',
    label: 'Medium Risk',
    shortLabel: 'Medium',
    rank: 3,
    description: 'Moderate commercial obligation or standard operational condition.',
    badgeBgLight: 'bg-amber-50',
    badgeTextLight: 'text-amber-800',
    badgeBorderLight: 'border-amber-200',
    badgeBgDark: 'dark:bg-amber-950/40',
    badgeTextDark: 'dark:text-amber-300',
    badgeBorderDark: 'dark:border-amber-900',
    highlightBgLight: 'bg-amber-100/60 hover:bg-amber-200/70',
    highlightBorderLight: 'border-b border-amber-500',
    highlightBgDark: 'dark:bg-amber-950/50 dark:hover:bg-amber-900/60',
    highlightBorderDark: 'dark:border-amber-400',
    iconName: 'Info'
  },
  LOW: {
    level: 'LOW',
    label: 'Low Risk',
    shortLabel: 'Low',
    rank: 2,
    description: 'Customary or mutual legal provision presenting standard commercial terms.',
    badgeBgLight: 'bg-emerald-50',
    badgeTextLight: 'text-emerald-800',
    badgeBorderLight: 'border-emerald-200',
    badgeBgDark: 'dark:bg-emerald-950/40',
    badgeTextDark: 'dark:text-emerald-300',
    badgeBorderDark: 'dark:border-emerald-900',
    highlightBgLight: 'bg-emerald-100/50 hover:bg-emerald-200/60',
    highlightBorderLight: 'border-b border-emerald-500',
    highlightBgDark: 'dark:bg-emerald-950/40 dark:hover:bg-emerald-900/50',
    highlightBorderDark: 'dark:border-emerald-400',
    iconName: 'CheckCircle'
  },
  NEEDS_REVIEW: {
    level: 'NEEDS_REVIEW',
    label: 'Needs Review',
    shortLabel: 'Needs Review',
    rank: 1,
    description: 'Ambiguous or incomplete clause requiring human verification.',
    badgeBgLight: 'bg-slate-100',
    badgeTextLight: 'text-slate-700',
    badgeBorderLight: 'border-slate-200',
    badgeBgDark: 'dark:bg-slate-800',
    badgeTextDark: 'dark:text-slate-300',
    badgeBorderDark: 'dark:border-slate-700',
    highlightBgLight: 'bg-slate-100 hover:bg-slate-200',
    highlightBorderLight: 'border-b border-dashed border-slate-500',
    highlightBgDark: 'dark:bg-slate-800/80 dark:hover:bg-slate-700',
    highlightBorderDark: 'dark:border-dashed dark:border-slate-400',
    iconName: 'HelpCircle'
  }
};

export const RISK_ORDER: RiskLevel[] = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'NEEDS_REVIEW'];

export function getRiskMeta(level: RiskLevel): RiskMetadata {
  return RISK_LEVELS[level] || RISK_LEVELS.NEEDS_REVIEW;
}

export function sortClausesByRisk(
  clauses: ExtractedClause[],
  order: 'desc' | 'asc' = 'desc'
): ExtractedClause[] {
  return [...clauses].sort((a, b) => {
    const rankA = getRiskMeta(a.risk_level).rank;
    const rankB = getRiskMeta(b.risk_level).rank;
    if (rankA !== rankB) {
      return order === 'desc' ? rankB - rankA : rankA - rankB;
    }
    // secondary sort by start_char in document
    return a.start_char - b.start_char;
  });
}

export function getHighestRisk(levels: RiskLevel[]): RiskLevel | null {
  if (!levels.length) return null;
  let highest = levels[0];
  for (const lvl of levels) {
    if (getRiskMeta(lvl).rank > getRiskMeta(highest).rank) {
      highest = lvl;
    }
  }
  return highest;
}

export function getRiskCounts(clauses: ExtractedClause[]): Record<RiskLevel, number> {
  const counts: Record<RiskLevel, number> = {
    CRITICAL: 0,
    HIGH: 0,
    MEDIUM: 0,
    LOW: 0,
    NEEDS_REVIEW: 0
  };
  for (const clause of clauses) {
    if (counts[clause.risk_level] !== undefined) {
      counts[clause.risk_level]++;
    } else {
      counts.NEEDS_REVIEW++;
    }
  }
  return counts;
}
