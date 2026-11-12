import React from 'react';
import {
  AlertOctagon,
  AlertTriangle,
  Info,
  CheckCircle,
  HelpCircle
} from 'lucide-react';
import type { RiskLevel } from '../../types/analysis';
import { getRiskMeta } from '../../lib/risk';
import { cn } from '../../lib/utils';

interface RiskBadgeProps {
  level: RiskLevel;
  size?: 'sm' | 'md' | 'lg';
  showIcon?: boolean;
  className?: string;
}

export const RiskBadge: React.FC<RiskBadgeProps> = ({
  level,
  size = 'md',
  showIcon = true,
  className
}) => {
  const meta = getRiskMeta(level);

  const getIcon = () => {
    const iconSize = size === 'sm' ? 'w-3 h-3' : size === 'lg' ? 'w-4 h-4' : 'w-3.5 h-3.5';
    switch (meta.level) {
      case 'CRITICAL':
        return <AlertOctagon className={cn(iconSize, 'shrink-0 text-red-700 dark:text-red-400')} aria-hidden="true" />;
      case 'HIGH':
        return <AlertTriangle className={cn(iconSize, 'shrink-0 text-orange-700 dark:text-orange-400')} aria-hidden="true" />;
      case 'MEDIUM':
        return <Info className={cn(iconSize, 'shrink-0 text-amber-700 dark:text-amber-400')} aria-hidden="true" />;
      case 'LOW':
        return <CheckCircle className={cn(iconSize, 'shrink-0 text-emerald-700 dark:text-emerald-400')} aria-hidden="true" />;
      case 'NEEDS_REVIEW':
      default:
        return <HelpCircle className={cn(iconSize, 'shrink-0 text-slate-600 dark:text-slate-400')} aria-hidden="true" />;
    }
  };

  const sizeClasses = {
    sm: 'text-[11px] px-2 py-0.5 gap-1',
    md: 'text-xs px-2.5 py-0.5 gap-1.5',
    lg: 'text-sm px-3 py-1 gap-2 font-medium'
  }[size];

  return (
    <span
      className={cn(
        'inline-flex items-center font-medium rounded border transition-colors',
        meta.badgeBgLight,
        meta.badgeTextLight,
        meta.badgeBorderLight,
        meta.badgeBgDark,
        meta.badgeTextDark,
        meta.badgeBorderDark,
        sizeClasses,
        className
      )}
      role="status"
      aria-label={`Risk Level: ${meta.label}`}
      title={meta.description}
    >
      {showIcon && getIcon()}
      <span>{meta.label}</span>
    </span>
  );
};
