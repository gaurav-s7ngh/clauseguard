import React, { useState, useMemo } from 'react';
import { useParams, Link } from 'react-router-dom';
import { ArrowLeft, Loader2, AlertCircle, FileText, ListFilter } from 'lucide-react';
import { useAnalysis } from '../hooks/useAnalysis';
import type { ClauseCategory, ExtractedClause, RiskLevel, ReviewStatus } from '../types/analysis';
import { SummaryHeader } from '../components/risk/SummaryHeader';
import { DocumentViewer } from '../components/viewer/DocumentViewer';
import { RiskFilters } from '../components/risk/RiskFilters';
import { RiskCardList } from '../components/risk/RiskCardList';
import { ExplanationDrawer } from '../components/risk/ExplanationDrawer';

export const ResultsPage: React.FC = () => {
  const { documentId } = useParams<{ documentId: string }>();
  const { data: result, isLoading, isError, error, refetch } = useAnalysis(documentId);

  // Active selected clause for two-way sync
  const [selectedClause, setSelectedClause] = useState<ExtractedClause | null>(null);

  // Review checklist statuses (clauseId -> ReviewStatus)
  const [reviewStatuses, setReviewStatuses] = useState<Record<string, ReviewStatus>>({});

  // Filters
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<ClauseCategory | 'ALL'>('ALL');
  const [selectedRisk, setSelectedRisk] = useState<RiskLevel | 'ALL'>('ALL');

  // Mobile layout tab: 'viewer' | 'clauses'
  const [mobileTab, setMobileTab] = useState<'viewer' | 'clauses'>('viewer');

  // Filter logic
  const rawClauses = result?.clauses;
  const filteredClauses = useMemo(() => {
    if (!rawClauses) return [];
    return rawClauses.filter((clause) => {
      // Category filter
      if (selectedCategory !== 'ALL' && clause.category !== selectedCategory) {
        return false;
      }
      // Risk filter
      if (selectedRisk !== 'ALL' && clause.risk_level !== selectedRisk) {
        return false;
      }
      // Text search
      if (searchQuery.trim().length > 0) {
        const q = searchQuery.toLowerCase();
        const textMatch = clause.clause_text.toLowerCase().includes(q);
        const explMatch = clause.plain_english_explanation.toLowerCase().includes(q);
        const reasonMatch = clause.risk_reasoning.toLowerCase().includes(q);
        if (!textMatch && !explMatch && !reasonMatch) {
          return false;
        }
      }
      return true;
    });
  }, [rawClauses, selectedCategory, selectedRisk, searchQuery]);

  const handleResetFilters = () => {
    setSearchQuery('');
    setSelectedCategory('ALL');
    setSelectedRisk('ALL');
  };

  const handleSelectClause = (clause: ExtractedClause) => {
    setSelectedClause(clause);
  };

  const handleStatusChange = (clauseId: string, status: ReviewStatus) => {
    setReviewStatuses((prev) => ({
      ...prev,
      [clauseId]: status
    }));
  };

  // Loading state
  if (isLoading) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center p-12 space-y-3">
        <Loader2 className="w-6 h-6 text-[#315C8A] animate-spin" />
        <div className="text-center space-y-1">
          <p className="text-sm font-medium text-[#202938] dark:text-[#F1F3F7]">
            Analyzing contract...
          </p>
          <p className="text-xs text-[#687385] dark:text-[#8B97A8]">
            Identifying clauses, checking risks and preparing plain-English explanations.
          </p>
        </div>
      </div>
    );
  }

  // Error state
  if (isError || !result) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center p-6">
        <div className="max-w-md w-full p-6 bg-white dark:bg-[#181D26] rounded border border-[#E4E7EC] dark:border-[#28303E] text-center space-y-3">
          <AlertCircle className="w-6 h-6 mx-auto text-red-600" />
          <h2 className="text-sm font-semibold text-[#202938] dark:text-[#F1F3F7]">
            Unable to load contract analysis
          </h2>
          <p className="text-xs text-[#687385] dark:text-[#8B97A8]">
            {error?.message || 'Could not retrieve analysis for this document ID.'}
          </p>
          <div className="flex justify-center gap-3 pt-2">
            <button
              onClick={() => refetch()}
              type="button"
              className="px-3 py-1.5 text-xs font-medium rounded bg-[#315C8A] text-white hover:bg-[#264A70]"
            >
              Try again
            </button>
            <Link
              to="/"
              className="px-3 py-1.5 text-xs font-medium rounded border border-[#D0D5DD] dark:border-[#384252] text-[#202938] dark:text-[#F1F3F7]"
            >
              Upload another file
            </Link>
          </div>
        </div>
      </div>
    );
  }

  const selectedClauseKey = selectedClause
    ? selectedClause.id || `${selectedClause.category}-${selectedClause.start_char}`
    : '';

  return (
    <div className="flex-1 flex flex-col p-4 sm:p-6 max-w-7xl mx-auto w-full space-y-4">
      {/* Back action */}
      <div className="flex items-center justify-between">
        <Link
          to="/"
          className="inline-flex items-center gap-1.5 text-xs text-[#687385] dark:text-[#8B97A8] hover:text-[#202938] dark:hover:text-white transition-colors"
        >
          <ArrowLeft className="w-3.5 h-3.5" />
          <span>Upload another document</span>
        </Link>
      </div>

      {/* Top Section: Summary Header */}
      <SummaryHeader result={result} />

      {/* Mobile Tab Switcher (<1024px) */}
      <div className="lg:hidden flex items-center p-1 bg-[#F1F3F6] dark:bg-[#202633] rounded">
        <button
          onClick={() => setMobileTab('viewer')}
          type="button"
          className={`flex-1 py-1.5 text-xs font-medium rounded flex items-center justify-center gap-1.5 transition-colors ${
            mobileTab === 'viewer'
              ? 'bg-white dark:bg-[#181D26] text-[#315C8A] dark:text-[#5D8EBF] shadow-xs'
              : 'text-[#687385] dark:text-[#8B97A8]'
          }`}
        >
          <FileText className="w-3.5 h-3.5" />
          <span>Document</span>
        </button>

        <button
          onClick={() => setMobileTab('clauses')}
          type="button"
          className={`flex-1 py-1.5 text-xs font-medium rounded flex items-center justify-center gap-1.5 transition-colors ${
            mobileTab === 'clauses'
              ? 'bg-white dark:bg-[#181D26] text-[#315C8A] dark:text-[#5D8EBF] shadow-xs'
              : 'text-[#687385] dark:text-[#8B97A8]'
          }`}
        >
          <ListFilter className="w-3.5 h-3.5" />
          <span>Clauses ({filteredClauses.length})</span>
        </button>
      </div>

      {/* Main Workspace: Two-column layout (65% Document / 35% Clauses) */}
      <div className="flex-1 grid grid-cols-1 lg:grid-cols-12 gap-5 min-h-[600px] items-stretch">
        {/* Left Side: Original Document (~65%, lg: 8 cols) */}
        <div
          className={`lg:col-span-8 flex flex-col min-h-[500px] ${
            mobileTab !== 'viewer' ? 'hidden lg:flex' : 'flex'
          }`}
        >
          <DocumentViewer
            documentText={result.document_text}
            clauses={filteredClauses}
            selectedClause={selectedClause}
            onSelectClause={handleSelectClause}
            className="h-[750px]"
          />
        </div>

        {/* Right Side: Detected Clauses (~35%, lg: 4 cols) */}
        <div
          className={`lg:col-span-4 flex flex-col space-y-3 ${
            mobileTab === 'viewer' ? 'hidden lg:flex' : 'flex'
          }`}
        >
          {/* Simple compact filters */}
          <RiskFilters
            searchQuery={searchQuery}
            onSearchChange={setSearchQuery}
            selectedCategory={selectedCategory}
            onCategoryChange={setSelectedCategory}
            selectedRisk={selectedRisk}
            onRiskChange={setSelectedRisk}
            onResetFilters={handleResetFilters}
            totalFilteredCount={filteredClauses.length}
            totalCount={result.clauses.length}
          />

          {/* Vertical Clauses List */}
          <div className="flex-1 overflow-y-auto custom-scrollbar max-h-[660px]">
            <RiskCardList
              clauses={filteredClauses}
              selectedClause={selectedClause}
              onSelectClause={handleSelectClause}
              reviewStatuses={reviewStatuses}
              onStatusChange={handleStatusChange}
            />
          </div>
        </div>
      </div>

      {/* Human-Centered Clause Details Panel */}
      <ExplanationDrawer
        clause={selectedClause}
        onClose={() => setSelectedClause(null)}
        reviewStatus={selectedClauseKey ? reviewStatuses[selectedClauseKey] : 'not_reviewed'}
        onStatusChange={handleStatusChange}
      />
    </div>
  );
};
