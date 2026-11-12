import { jsPDF } from 'jspdf';
import type { AnalysisResult } from '../types/analysis';
import { getRiskCounts, sortClausesByRisk } from './risk';
import { getCategoryMeta } from './categories';
import { formatConfidence } from './utils';

export function exportAnalysisReportPDF(result: AnalysisResult): void {
  const doc = new jsPDF({
    orientation: 'portrait',
    unit: 'pt',
    format: 'a4'
  });

  const pageWidth = doc.internal.pageSize.getWidth();
  const pageHeight = doc.internal.pageSize.getHeight();
  const margin = 40;
  const contentWidth = pageWidth - margin * 2;
  let cursorY = margin;

  function checkPageBreak(spaceNeeded: number) {
    if (cursorY + spaceNeeded > pageHeight - margin - 40) {
      addFooter();
      doc.addPage();
      cursorY = margin + 20;
    }
  }

  function addFooter() {
    doc.setFont('helvetica', 'normal');
    doc.setFontSize(8);
    doc.setTextColor(120, 120, 120);
    const disclaimer =
      'ClauseGuard is an educational document-analysis tool. It does not provide legal advice or replace consultation with a qualified legal professional.';
    doc.text(disclaimer, margin, pageHeight - 25, { maxWidth: contentWidth });
  }

  // --- Document Header ---
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(18);
  doc.setTextColor(24, 24, 27);
  doc.text('CLAUSEGUARD LEGAL RISK ASSESSMENT REPORT', margin, cursorY);
  cursorY += 24;

  // Metadata subheader
  doc.setFont('helvetica', 'normal');
  doc.setFontSize(9);
  doc.setTextColor(80, 80, 80);
  const dateStr = new Date().toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'long',
    day: 'numeric'
  });
  doc.text(`Document: ${result.filename}  |  Generated: ${dateStr}`, margin, cursorY);
  cursorY += 15;
  doc.setDrawColor(210, 210, 215);
  doc.line(margin, cursorY, margin + contentWidth, cursorY);
  cursorY += 20;

  // --- Risk Counts Summary Cards ---
  const counts = getRiskCounts(result.clauses);
  const cardWidth = (contentWidth - 4 * 8) / 5;
  const cardHeight = 44;

  const riskTiers = [
    { label: 'CRITICAL', count: counts.CRITICAL, color: [225, 29, 72] }, // Rose
    { label: 'HIGH', count: counts.HIGH, color: [217, 119, 6] },        // Amber
    { label: 'MEDIUM', count: counts.MEDIUM, color: [202, 138, 4] },    // Yellow
    { label: 'LOW', count: counts.LOW, color: [16, 185, 129] },         // Emerald
    { label: 'NEEDS REVIEW', count: counts.NEEDS_REVIEW, color: [100, 116, 139] } // Slate
  ];

  riskTiers.forEach((tier, index) => {
    const x = margin + index * (cardWidth + 8);
    doc.setFillColor(tier.color[0], tier.color[1], tier.color[2]);
    doc.roundedRect(x, cursorY, cardWidth, 4, 2, 2, 'F');

    doc.setFillColor(248, 250, 252);
    doc.roundedRect(x, cursorY + 4, cardWidth, cardHeight - 4, 2, 2, 'F');
    doc.setDrawColor(226, 232, 240);
    doc.roundedRect(x, cursorY, cardWidth, cardHeight, 2, 2, 'S');

    doc.setFont('helvetica', 'bold');
    doc.setFontSize(14);
    doc.setTextColor(24, 24, 27);
    doc.text(String(tier.count), x + cardWidth / 2, cursorY + 22, { align: 'center' });

    doc.setFont('helvetica', 'normal');
    doc.setFontSize(7.5);
    doc.setTextColor( tier.color[0], tier.color[1], tier.color[2]);
    doc.text(tier.label, x + cardWidth / 2, cursorY + 36, { align: 'center' });
  });

  cursorY += cardHeight + 25;

  // --- Extracted Clauses Section ---
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(13);
  doc.setTextColor(24, 24, 27);
  doc.text(`Identified Clauses & Detailed Assessments (${result.clauses.length} Total)`, margin, cursorY);
  cursorY += 16;

  const sortedClauses = sortClausesByRisk(result.clauses, 'desc');

  for (let i = 0; i < sortedClauses.length; i++) {
    const clause = sortedClauses[i];
    const catMeta = getCategoryMeta(clause.category);

    checkPageBreak(120);

    // Clause Item Box
    const boxStartY = cursorY;
    doc.setFillColor(255, 255, 255);
    doc.setDrawColor(226, 232, 240);

    // Category and Risk Header
    cursorY += 14;
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(11);
    doc.setTextColor(15, 23, 42);
    doc.text(`${i + 1}. ${catMeta.name}`, margin + 8, cursorY);

    // Risk badge text
    doc.setFontSize(9);
    if (clause.risk_level === 'CRITICAL') doc.setTextColor(190, 18, 60);
    else if (clause.risk_level === 'HIGH') doc.setTextColor(180, 83, 9);
    else if (clause.risk_level === 'MEDIUM') doc.setTextColor(161, 98, 7);
    else if (clause.risk_level === 'LOW') doc.setTextColor(4, 120, 87);
    else doc.setTextColor(71, 85, 105);

    const riskBadge = `[${clause.risk_level}]`;
    const headerRightText = `${riskBadge}   Detection Confidence: ${formatConfidence(clause.confidence)}`;
    doc.text(headerRightText, margin + contentWidth - 8, cursorY, { align: 'right' });
    cursorY += 12;

    // Clause text quote
    doc.setFont('helvetica', 'italic');
    doc.setFontSize(8.5);
    doc.setTextColor(75, 85, 99);
    const clauseLines = doc.splitTextToSize(`"${clause.clause_text}"`, contentWidth - 16);
    checkPageBreak(clauseLines.length * 10 + 60);
    doc.text(clauseLines, margin + 8, cursorY);
    cursorY += clauseLines.length * 10 + 8;

    // Plain English Explanation
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(8.5);
    doc.setTextColor(30, 41, 59);
    doc.text('What this means:', margin + 8, cursorY);
    cursorY += 10;

    doc.setFont('helvetica', 'normal');
    doc.setFontSize(8.5);
    doc.setTextColor(51, 65, 85);
    const expLines = doc.splitTextToSize(clause.plain_english_explanation, contentWidth - 16);
    checkPageBreak(expLines.length * 10 + 40);
    doc.text(expLines, margin + 8, cursorY);
    cursorY += expLines.length * 10 + 6;

    // Risk Reasoning
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(8.5);
    doc.setTextColor(30, 41, 59);
    doc.text('Why it\'s flagged:', margin + 8, cursorY);
    cursorY += 10;

    doc.setFont('helvetica', 'normal');
    doc.setFontSize(8.5);
    doc.setTextColor(51, 65, 85);
    const reasonLines = doc.splitTextToSize(clause.risk_reasoning, contentWidth - 16);
    checkPageBreak(reasonLines.length * 10 + 30);
    doc.text(reasonLines, margin + 8, cursorY);
    cursorY += reasonLines.length * 10 + 6;

    // Suggested modification (if available)
    if (clause.suggested_modification) {
      doc.setFont('helvetica', 'bold');
      doc.setFontSize(8.5);
      doc.setTextColor(3, 105, 161);
      doc.text('Suggested Redline / Counter-Proposal:', margin + 8, cursorY);
      cursorY += 10;

      doc.setFont('courier', 'normal');
      doc.setFontSize(8);
      doc.setTextColor(30, 41, 59);
      const modLines = doc.splitTextToSize(clause.suggested_modification, contentWidth - 16);
      checkPageBreak(modLines.length * 9.5 + 20);
      doc.text(modLines, margin + 8, cursorY);
      cursorY += modLines.length * 9.5 + 6;
    }

    // Draw box surrounding the clause item
    const itemHeight = cursorY - boxStartY + 4;
    doc.roundedRect(margin, boxStartY, contentWidth, itemHeight, 3, 3, 'S');
    cursorY += 14;
  }

  // Final footer on last page
  addFooter();

  // Save / trigger browser download
  const cleanName = result.filename.replace(/[^a-zA-Z0-9_-]/g, '_');
  doc.save(`ClauseGuard_Report_${cleanName}.pdf`);
}

