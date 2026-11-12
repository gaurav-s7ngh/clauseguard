import type {
  AnalysisResult,
  AnalysisStatusResponse,
  ClauseCategory,
  DocumentHistoryItem,
  ExtractedClause,
  HealthCheckResponse,
  RiskLevel,
  UploadResponse
} from '../../types/analysis';
import type { ClauseGuardApiClient, RawClauseDTO } from './types';
import { adaptAnalysisResult } from './adapters';

export const SAMPLE_CONTRACT_TEXT = `MASTER CLOUD SERVICES AND PLATFORM SUBSCRIPTION AGREEMENT

This Master Cloud Services and Platform Subscription Agreement (the "Agreement") is entered into and made effective as of October 14, 2026 (the "Effective Date"), by and between ApexCore Enterprise Technologies Inc., a Delaware corporation having its principal place of business at 100 Enterprise Way, Suite 400, Wilmington, DE 19801 ("Licensor" or "Vendor"), and the subscriber or commercial entity entering into or accepting this Agreement ("Customer" or "Licensee"). Licensor and Customer may each individually be referred to as a "Party" and collectively as the "Parties."

RECITALS

WHEREAS, Licensor has designed, developed, and commercializes a proprietary cloud-based workflow intelligence, automated document processing, and enterprise resource governance platform (the "Cloud Services"); and

WHEREAS, Customer desires to procure a subscription to access and utilize the Cloud Services for its internal corporate operations, subject to the strict covenants, operational limitations, risk allocations, and licensing terms set forth in this Agreement;

NOW, THEREFORE, in consideration of the mutual covenants, conditions, and promises contained herein and other good and valuable consideration, the receipt and legal sufficiency of which are hereby acknowledged, the Parties agree as follows:

SECTION 1. DEFINITIONS AND USAGE

1.1 "Authorized Users" means Customer's employees, direct contractors, and designated agents authorized by Customer to access the Cloud Services solely on Customer's behalf and strictly subject to user credential controls.
1.2 "Customer Data" means all electronic data, files, text, queries, or confidential documents submitted by or on behalf of Customer to the Cloud Services for processing.
1.3 "Documentation" means the official online user manuals, release notes, and technical specifications provided by Licensor to Customer describing the functional operation of the Cloud Services.
1.4 "Intellectual Property Rights" means all patent rights, copyrights, mask work rights, moral rights, trademark rights, trade secret rights, and all other proprietary rights recognized throughout the world.
1.5 "Term" means the Initial Term together with all subsequent Renewal Terms as defined in Section 8 herein.

SECTION 2. SUBSCRIPTION LICENSE GRANT AND RESTRICTIONS

2.1 Scope of Grant. Subject to the terms and conditions of this Agreement, Licensor hereby grants to Customer a limited, non-exclusive, non-transferable, revocable license during the Term to access and use the Cloud Software solely for Customer's internal business operations. Customer shall not sublicense, lease, resell, distribute, or provide time-sharing access to the Cloud Services to any third party.
2.2 Usage Limits. Customer shall strictly observe the monthly transaction volumes, storage thresholds, and named-seat maximums designated in the applicable Order Schedule. Any usage exceeding these prescribed tiers shall incur overage charges billed at Licensor's then-current standard commercial rates.

SECTION 3. INTELLECTUAL PROPERTY RIGHTS AND OWNERSHIP

3.1 Proprietary Ownership of Licensor. Customer expressly acknowledges that Licensor and its licensors retain all right, title, and interest in and to the Cloud Services, underlying algorithms, source code, data schemas, and user interfaces.
3.2 Customer Work Product and Derivative Works Assignment. Customer hereby irrevocably assigns, transfers, and conveys to Licensor all right, title, and interest worldwide in and to all derivative works, customizations, modifications, integrations, feedback, and work product developed in connection with the Software, whether developed solely by Customer or jointly with Licensor. Customer waives all moral rights and agrees to execute any documentation necessary to confirm Licensor's sole title thereto.
3.3 Customer Data Ownership. Licensor acknowledges that Customer retains exclusive ownership of Customer Data, subject to the operational licenses granted to Licensor to store, process, and anonymize such data for platform maintenance.

SECTION 4. FEES, INVOICING, AND PAYMENT TERMS

4.1 Subscription Fees. Customer shall pay all subscription fees, implementation charges, and auxiliary service costs specified in each executed Order Schedule. All fees are quoted and payable in United States Dollars (USD).
4.2 Payment Timeline. All undisputed invoices are due and payable net thirty (30) days from the date of invoice. Late payments shall accrue interest at the rate of one and one-half percent (1.5%) per month or the highest statutory rate permissible under applicable law.
4.3 Taxes. Fees stated herein are exclusive of applicable value-added, sales, use, or withholding taxes, which shall be the sole responsibility of Customer.

SECTION 5. REPRESENTATIONS, WARRANTIES, AND DISCLAIMERS

5.1 Mutual Authority. Each Party represents and warrants that it has full corporate power and authority to enter into and perform its obligations under this Agreement.
5.2 Warranty Disclaimer. EXCEPT AS EXPRESSLY SET FORTH HEREIN, THE CLOUD SERVICES, DOCUMENTATION, AND ALL ANCILLARY SERVICES ARE PROVIDED "AS IS" AND "AS AVAILABLE" WITHOUT WARRANTY OF ANY KIND, EXPRESS, IMPLIED, STATUTORY, OR OTHERWISE, INCLUDING WITHOUT LIMITATION WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NON-INFRINGEMENT.

SECTION 6. LIMITATION OF MONETARY LIABILITY

6.1 Aggregate Dollar Limitation. IN NO EVENT SHALL LICENSOR'S AGGREGATE LIABILITY ARISING OUT OF OR RELATED TO THIS AGREEMENT, REGARDLESS OF THE THEORY OF LIABILITY, EXCEED THE NOMINAL SUM OF ONE HUNDRED DOLLARS ($100.00). CUSTOMER EXPRESSLY WAIVES ANY CLAIM FOR CONSEQUENTIAL, SPECIAL, PUNITIVE, OR INDIRECT DAMAGES.
6.2 Carve-outs and Exclusions. Notwithstanding the foregoing, neither party's liability shall be limited with respect to gross negligence, intentional misconduct, or direct breach of Section 7 (Confidentiality).
6.3 Risk Allocation Acknowledgement. The Parties agree that the limitations set forth in this Section 6 reflect an intentional allocation of risk between the Parties and that Licensor would not enter into this Agreement without such limitations.

SECTION 7. CONFIDENTIALITY AND SECURITY SAFEGUARDS

7.1 Definition of Confidential Information. "Confidential Information" means all non-public, proprietary information disclosed by one Party to the other Party, whether orally or in writing, that is designated as confidential or that reasonably should be understood to be confidential.
7.2 Duty of Care. Each Party agrees to protect the Confidential Information of the other Party with the same degree of care it uses to protect its own confidential materials of like nature, but in no event less than reasonable care.

SECTION 8. TERM, RENEWAL, AND TERMINATION FOR CONVENIENCE

8.1 Initial Term. The Initial Term of this Agreement shall commence on the Effective Date and continue for a period of twelve (12) calendar months.
8.2 Automatic Renewal Traps. This Agreement shall automatically renew for successive terms of twenty-four (24) months each, unless Customer provides formal written notice of non-renewal at least ninety (90) days prior to the expiration of the then-current term. Upon each automatic renewal, Licensor reserves the right to increase subscription fees by fifteen percent (15%).
8.3 Termination for Convenience. Licensor reserves the unilateral right to suspend or terminate this Agreement and Customer's access to the Service at any time, for any reason or no reason, immediately upon written or electronic notice without penalty or liability. In the event of early termination by Customer, Customer shall pay an early exit penalty equal to the remaining fees for the unexpired Term.
8.4 Termination for Cause. Either Party may terminate this Agreement upon thirty (30) days' written notice if the other Party materially breaches any provision and fails to cure such breach within the cure window.

SECTION 9. AUDIT RIGHTS, INSPECTIONS, AND COMPLIANCE

9.1 On-Site Inspection Access. Licensor or its designated independent auditor shall have the right, without prior written notice, during normal business hours to enter Customer's facilities and inspect Customer's books, records, computing environments, and network logs to verify compliance with this Agreement. If any discrepancy is discovered, Customer shall promptly reimburse Licensor for the full cost of the audit.
9.2 Supplementary Audit Verification. Additional verification protocols and compliance benchmarks shall be determined in accordance with the Security Exhibit referenced in Section 14.

SECTION 10. NON-COMPETE COVENANTS AND RESTRICTIVE OBLIGATIONS

10.1 Restrictive Scope and Duration. During the Term and for a period of twenty-four (24) months following termination or expiration of this Agreement for any reason, Customer covenants and agrees that it shall not, directly or indirectly, develop, market, license, distribute, or invest in any software, service, or product that competes with or performs substantially similar functions to the Cloud Services.
10.2 Non-Solicitation. During the Term and for twelve (12) months thereafter, Customer shall not solicit, recruit, or attempt to hire any engineer or employee of Licensor without Licensor's prior written permission.

SECTION 11. ANTI-ASSIGNMENT AND SUCCESSION CONTROLS

11.1 Prohibition on Assignment. Customer may not assign, delegate, or transfer this Agreement, or any of its rights or obligations hereunder, in whole or in part, by operation of law, merger, change of control, or otherwise, without the prior express written consent of Licensor, which consent Licensor may withhold in its sole and absolute discretion. Any attempted assignment in violation of this Section 11 shall be null and void ab initio.
11.2 Licensor Assignment. Licensor may freely assign or transfer this Agreement and its rights and obligations to any affiliate, subsidiary, or successor entity in connection with any merger, acquisition, corporate reorganization, or sale of assets.

SECTION 12. GOVERNING LAW, FORUM, AND JURISDICTION

12.1 Governing Law. This Agreement and all disputes arising out of or related to this Agreement shall be governed by, and construed in accordance with, the substantive laws of the State of Delaware, without giving effect to any choice of law rules. The parties irrevocably submit to the exclusive jurisdiction of the state and federal courts located in Wilmington, Delaware.
12.2 Waiver of Jury Trial. TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, EACH PARTY HEREBY IRREVOCABLY WAIVES ALL RIGHTS TO A TRIAL BY JURY IN ANY ACTION, PROCEEDING, OR COUNTERCLAIM ARISING OUT OF THIS AGREEMENT.

SECTION 13. INDEMNIFICATION AND REMEDIES

13.1 Customer Indemnification. Customer shall defend, indemnify, and hold harmless Licensor and its officers, directors, employees, and agents from and against any third-party claims, liabilities, losses, damages, and expenses arising out of Customer's use of the Cloud Services or violation of Section 2 or Section 10.
13.2 Equitable Relief. Customer acknowledges that any breach of Section 3, Section 7, or Section 10 may cause irreparable harm for which monetary damages alone would be inadequate, entitling Licensor to seek immediate injunctive relief without the requirement of posting a bond.

SECTION 14. GENERAL PROVISIONS

14.1 Severability. If any provision of this Agreement is held by a court of competent jurisdiction to be contrary to law, such provision shall be modified to the minimum extent necessary to make it valid, and the remaining provisions shall continue in full force.
14.2 Entire Agreement. This Agreement, together with all schedules, exhibits, and order forms, constitutes the final and complete understanding between the Parties concerning the subject matter hereof, superseding all prior oral or written discussions, proposals, and understandings.
14.3 Amendments. No modification or waiver of any provision of this Agreement shall be effective unless set forth in a written instrument signed by duly authorized representatives of both Parties.

IN WITNESS WHEREOF, the Parties have caused this Master Cloud Services and Platform Subscription Agreement to be executed by their respective duly authorized officers as of the Effective Date.`;

/**
 * Helper to compute character offsets directly from exact substring
 */
function createClauseFromText(params: {
  category: ClauseCategory;
  clauseText: string;
  riskLevel: RiskLevel;
  confidence: number;
  riskScore?: number;
  riskReasoning: string;
  plainEnglishExplanation: string;
  suggestedModification?: string;
  documentId?: string;
}): ExtractedClause {
  const startChar = SAMPLE_CONTRACT_TEXT.indexOf(params.clauseText);
  if (startChar === -1) {
    throw new Error(
      `Mock configuration error: Clause text not found in SAMPLE_CONTRACT_TEXT: "${params.clauseText.slice(0, 40)}..."`
    );
  }
  const endChar = startChar + params.clauseText.length;

  return {
    id: `mock-${params.category}-${startChar}`,
    category: params.category,
    clause_text: params.clauseText,
    start_char: startChar,
    end_char: endChar,
    confidence: params.confidence,
    chunk_id: `chunk_${Math.floor(startChar / 1000)}`,
    document_id: params.documentId || 'doc_mock_sample_master',
    page_number: Math.floor(startChar / 3000) + 1,
    surrounding_context: SAMPLE_CONTRACT_TEXT.slice(
      Math.max(0, startChar - 80),
      Math.min(SAMPLE_CONTRACT_TEXT.length, endChar + 80)
    ),
    model_version: 'roberta-span-v1',
    extraction_status: 'extracted',
    risk_level: params.riskLevel,
    risk_score: params.riskScore,
    risk_reasoning: params.riskReasoning,
    plain_english_explanation: params.plainEnglishExplanation,
    suggested_modification: params.suggestedModification
  };
}

export function getMockSampleClauses(): ExtractedClause[] {
  return [
    createClauseFromText({
      category: 'license_grant',
      clauseText:
        'Subject to the terms and conditions of this Agreement, Licensor hereby grants to Customer a limited, non-exclusive, non-transferable, revocable license during the Term to access and use the Cloud Software solely for Customer\'s internal business operations.',
      riskLevel: 'MEDIUM',
      confidence: 0.94,
      riskScore: 5,
      riskReasoning:
        'The license grant is revocable and non-transferable, allowing the vendor broad discretion to suspend operations if disputes arise.',
      plainEnglishExplanation:
        'You are granted permission to use the software for your internal team only. The vendor can revoke or suspend this license under specific contractual conditions.',
      suggestedModification:
        'Subject to the terms of this Agreement, Licensor grants Customer an irrevocable (except for uncured material breach), non-exclusive license during the Term to access and use the Cloud Software.'
    }),
    createClauseFromText({
      category: 'ip_ownership_assignment',
      clauseText:
        'Customer hereby irrevocably assigns, transfers, and conveys to Licensor all right, title, and interest worldwide in and to all derivative works, customizations, modifications, integrations, feedback, and work product developed in connection with the Software, whether developed solely by Customer or jointly with Licensor.',
      riskLevel: 'CRITICAL',
      confidence: 0.97,
      riskScore: 10,
      riskReasoning:
        'Extreme IP forfeiture: Customer involuntarily transfers ownership of custom integrations, scripts, and derivative works to the vendor, even when created solely by Customer.',
      plainEnglishExplanation:
        'If you build custom scripts, connectors, or modifications to work with this software, this clause gives full legal ownership of your creations to the vendor.',
      suggestedModification:
        'Customer retains all right, title, and interest in and to Customer-created workflows, scripts, and integrations. Licensor retains ownership of its core platform.'
    }),
    createClauseFromText({
      category: 'liability_cap',
      clauseText:
        'IN NO EVENT SHALL LICENSOR\'S AGGREGATE LIABILITY ARISING OUT OF OR RELATED TO THIS AGREEMENT, REGARDLESS OF THE THEORY OF LIABILITY, EXCEED THE NOMINAL SUM OF ONE HUNDRED DOLLARS ($100.00). CUSTOMER EXPRESSLY WAIVES ANY CLAIM FOR CONSEQUENTIAL, SPECIAL, PUNITIVE, OR INDIRECT DAMAGES.',
      riskLevel: 'CRITICAL',
      confidence: 0.96,
      riskScore: 9,
      riskReasoning:
        'Nominal $100 liability ceiling. In the event of catastrophic data loss, platform outage, or breach of security by vendor, customer recovery is practically zero.',
      plainEnglishExplanation:
        'The vendor caps its maximum payout to just $100, no matter how much financial harm, data loss, or downtime their system causes your company.',
      suggestedModification:
        'In no event shall either party\'s total aggregate liability exceed the total fees paid or payable by Customer under this Agreement in the twelve (12) months preceding the claim.'
    }),
    createClauseFromText({
      category: 'liability_cap',
      clauseText:
        'Notwithstanding the foregoing, neither party\'s liability shall be limited with respect to gross negligence, intentional misconduct, or direct breach of Section 7 (Confidentiality).',
      riskLevel: 'LOW',
      confidence: 0.88,
      riskScore: 2,
      riskReasoning:
        'Standard mutual carve-out ensuring both parties remain accountable for severe misconduct and confidentiality leaks.',
      plainEnglishExplanation:
        'Neither company can hide behind liability limits if they act with gross negligence or breach confidential business secrets.',
      suggestedModification: undefined
    }),
    createClauseFromText({
      category: 'renewal_term',
      clauseText:
        'This Agreement shall automatically renew for successive terms of twenty-four (24) months each, unless Customer provides formal written notice of non-renewal at least ninety (90) days prior to the expiration of the then-current term. Upon each automatic renewal, Licensor reserves the right to increase subscription fees by fifteen percent (15%).',
      riskLevel: 'HIGH',
      confidence: 0.95,
      riskScore: 8,
      riskReasoning:
        'Excessive 2-year renewal lock-in coupled with a mandatory 90-day cancellation window and unilateral 15% price hike.',
      plainEnglishExplanation:
        'If you forget to send a cancellation letter 90 days before the contract ends, you are automatically locked into paying 15% more for another full two years.',
      suggestedModification:
        'This Agreement shall renew for successive 12-month periods unless either party gives written notice at least thirty (30) days prior. Any price adjustments shall not exceed 3% annually.'
    }),
    createClauseFromText({
      category: 'termination_for_convenience',
      clauseText:
        'Licensor reserves the unilateral right to suspend or terminate this Agreement and Customer\'s access to the Service at any time, for any reason or no reason, immediately upon written or electronic notice without penalty or liability. In the event of early termination by Customer, Customer shall pay an early exit penalty equal to the remaining fees for the unexpired Term.',
      riskLevel: 'HIGH',
      confidence: 0.93,
      riskScore: 8,
      riskReasoning:
        'Asymmetric termination rights: Vendor can drop Customer on zero notice, while Customer faces punitive financial penalties for leaving early.',
      plainEnglishExplanation:
        'The vendor can cut off your service at any time without reason, but if you want to leave, you must pay out the rest of the contract.',
      suggestedModification:
        'Either party may terminate this Agreement for convenience upon sixty (60) days\' prior written notice without penalty.'
    }),
    createClauseFromText({
      category: 'audit_rights',
      clauseText:
        'Licensor or its designated independent auditor shall have the right, without prior written notice, during normal business hours to enter Customer\'s facilities and inspect Customer\'s books, records, computing environments, and network logs to verify compliance with this Agreement. If any discrepancy is discovered, Customer shall promptly reimburse Licensor for the full cost of the audit.',
      riskLevel: 'HIGH',
      confidence: 0.91,
      riskScore: 7,
      riskReasoning:
        'Unannounced on-site physical and network inspection rights with unilateral cost shifting onto Customer.',
      plainEnglishExplanation:
        'Vendor can show up at your office unannounced to inspect your servers, logs, and financial records, and charge you the full audit bill.',
      suggestedModification:
        'Upon at least thirty (30) business days\' prior written notice, Licensor may conduct an annual remote audit during normal hours. Customer shall only pay audit costs if an underpayment exceeding 5% is identified.'
    }),
    createClauseFromText({
      category: 'audit_rights',
      clauseText:
        'Additional verification protocols and compliance benchmarks shall be determined in accordance with the Security Exhibit referenced in Section 14.',
      riskLevel: 'NEEDS_REVIEW',
      confidence: 0.72,
      riskScore: undefined,
      riskReasoning:
        'External document dependency: Terms refer to an unattached Security Exhibit whose compliance requirements could not be verified automatically.',
      plainEnglishExplanation:
        'The automated system detected an audit provision, but its exact standards are tucked away in a separate exhibit. Manual verification is required.',
      suggestedModification:
        'Ensure the referenced Security Exhibit is provided and reviewed before signing.'
    }),
    createClauseFromText({
      category: 'non_compete',
      clauseText:
        'During the Term and for a period of twenty-four (24) months following termination or expiration of this Agreement for any reason, Customer covenants and agrees that it shall not, directly or indirectly, develop, market, license, distribute, or invest in any software, service, or product that competes with or performs substantially similar functions to the Cloud Services.',
      riskLevel: 'CRITICAL',
      confidence: 0.98,
      riskScore: 9,
      riskReasoning:
        'Broad 2-year non-compete binding Customer from developing or investing in competing software or tools.',
      plainEnglishExplanation:
        'You agree not to build, sell, or invest in any software that competes with this vendor for two full years after stopping use of their product.',
      suggestedModification:
        'Strike Section 10.1 in its entirety. Standard non-disclosure of proprietary trade secrets should suffice.'
    }),
    createClauseFromText({
      category: 'anti_assignment',
      clauseText:
        'Customer may not assign, delegate, or transfer this Agreement, or any of its rights or obligations hereunder, in whole or in part, by operation of law, merger, change of control, or otherwise, without the prior express written consent of Licensor, which consent Licensor may withhold in its sole and absolute discretion.',
      riskLevel: 'MEDIUM',
      confidence: 0.89,
      riskScore: 6,
      riskReasoning:
        'Blocks corporate mergers, acquisitions, or restructuring without vendor\'s discretionary written consent.',
      plainEnglishExplanation:
        'If your company is acquired, merges, or restructures, you cannot transfer this contract without the vendor\'s express permission, which they can withhold for any reason.',
      suggestedModification:
        'Neither party may assign this Agreement without consent, except that either party may assign to an affiliate or successor in connection with a merger, reorganization, or sale of substantially all assets.'
    }),
    createClauseFromText({
      category: 'governing_law',
      clauseText:
        'This Agreement and all disputes arising out of or related to this Agreement shall be governed by, and construed in accordance with, the substantive laws of the State of Delaware, without giving effect to any choice of law rules. The parties irrevocably submit to the exclusive jurisdiction of the state and federal courts located in Wilmington, Delaware.',
      riskLevel: 'LOW',
      confidence: 0.95,
      riskScore: 2,
      riskReasoning:
        'Standard commercial governing law and forum selection (Delaware corporate law). Predictable legal framework.',
      plainEnglishExplanation:
        'Any legal disputes will be resolved under Delaware law in Delaware courts, which is standard for corporate agreements.',
      suggestedModification: undefined
    })
  ];
}

export function getMockAnalysisResult(documentId = 'doc_mock_sample_master'): AnalysisResult {
  const clauses = getMockSampleClauses();
  return adaptAnalysisResult({
    document_id: documentId,
    filename: 'Master_Cloud_Services_Agreement_2026.txt',
    document_text: SAMPLE_CONTRACT_TEXT,
    document_char_length: SAMPLE_CONTRACT_TEXT.length,
    total_chunks: 14,
    total_extracted_clauses: clauses.length,
    pipeline_version: 'clauseguard-stage4-v1.2 (Mock Engine)',
    clauses: clauses
  });
}

/**
 * Mock API Client implementing ClauseGuardApiClient interface
 */
function extractDynamicClausesFromText(text: string, docId: string): RawClauseDTO[] {
  const patterns: Array<{
    category: string;
    regex: RegExp;
    risk_level: string;
    risk_score: number;
    explanation: string;
    why_it_matters: string;
    suggestion?: string;
  }> = [
    {
      category: 'liability_cap',
      regex: /(?:in\s+no\s+event\s+shall|limitation\s+of\s+liability|aggregate\s+liability|maximum\s+liability|liability\s+(?:shall|is|will\s+not)\s+exceed|consequential\s+damages)[^.\n]+(?:\.[^.\n]+)?/i,
      risk_level: 'CRITICAL',
      risk_score: 9,
      explanation: 'Caps the total monetary compensation or excludes indirect damages.',
      why_it_matters: 'If the other party causes major financial harm, you may not be able to recover your losses.',
      suggestion: 'Ensure the cap is mutual and excludes breaches of confidentiality or gross negligence.'
    },
    {
      category: 'termination_for_convenience',
      regex: /(?:terminate\s+(?:this\s+agreement\s+)?(?:at\s+any\s+time\s+)?without\s+cause|termination\s+for\s+convenience|upon\s+\d+\s+days['’]?\s+(?:prior\s+)?written\s+notice)[^.\n]+(?:\.[^.\n]+)?/i,
      risk_level: 'HIGH',
      risk_score: 8,
      explanation: 'Allows ending the contract early without needing to prove a breach or wrongdoing.',
      why_it_matters: 'The contract can be canceled abruptly with minimal warning, impacting planning and cash flow.',
      suggestion: 'Require at least 30 to 60 days written notice and reimbursement for unrecoverable expenses.'
    },
    {
      category: 'anti_assignment',
      regex: /(?:neither\s+party\s+may\s+assign|shall\s+not\s+assign|assignment\s+without\s+(?:the\s+)?prior\s+written\s+consent|no\s+assignment)[^.\n]+(?:\.[^.\n]+)?/i,
      risk_level: 'MEDIUM',
      risk_score: 6,
      explanation: 'Restricts transferring contract rights or obligations to a third party or buyer.',
      why_it_matters: 'Can block company acquisition, corporate restructuring, or assigning invoices.',
      suggestion: 'Add an exception allowing assignment in connection with a merger or sale of substantially all assets.'
    },
    {
      category: 'renewal_term',
      regex: /(?:automatically\s+renew|automatic\s+renewal|renewal\s+term|successive\s+(?:terms|periods)|evergreen)[^.\n]+(?:\.[^.\n]+)?/i,
      risk_level: 'HIGH',
      risk_score: 8,
      explanation: 'The contract automatically extends for another term unless canceled in advance.',
      why_it_matters: 'Missing the cancellation window locks you into paying for another full term.',
      suggestion: 'Set a calendar reminder before the non-renewal notice deadline.'
    },
    {
      category: 'governing_law',
      regex: /(?:governed\s+by\s+and\s+construed\s+in\s+accordance\s+with|choice\s+of\s+law|jurisdiction\s+and\s+venue|laws\s+of\s+the\s+state\s+of)[^.\n]+(?:\.[^.\n]+)?/i,
      risk_level: 'LOW',
      risk_score: 3,
      explanation: 'Determines which state or country laws apply if a legal dispute arises.',
      why_it_matters: 'Resolving disputes in distant courts increases travel and litigation costs.',
      suggestion: 'Prefer your home state jurisdiction or mutual neutral arbitration.'
    },
    {
      category: 'audit_rights',
      regex: /(?:right\s+to\s+audit|audit\s+rights|inspect\s+(?:the\s+)?books\s+and\s+records|audit\s+and\s+inspect)[^.\n]+(?:\.[^.\n]+)?/i,
      risk_level: 'MEDIUM',
      risk_score: 6,
      explanation: 'Allows one party to inspect systems, accounting records, or facilities.',
      why_it_matters: 'Unrestricted audits can be disruptive and expose confidential information.',
      suggestion: 'Limit audits to once per year, during normal business hours, with at least 15 days notice.'
    },
    {
      category: 'ip_ownership_assignment',
      regex: /(?:work\s+made\s+for\s+hire|hereby\s+assigns\s+all\s+right|all\s+right,\s*title\s+and\s+interest|sole\s+and\s+exclusive\s+property)[^.\n]+(?:\.[^.\n]+)?/i,
      risk_level: 'CRITICAL',
      risk_score: 9,
      explanation: 'Transfers ownership of all created materials, designs, or code to the other party.',
      why_it_matters: 'You may lose rights to your own background tools, pre-existing IP, and templates.',
      suggestion: 'Retain ownership of your pre-existing tools and grant a license instead.'
    },
    {
      category: 'license_grant',
      regex: /(?:hereby\s+grants?\s+(?:a\s+)?(?:non-exclusive|exclusive|worldwide|royalty-free|perpetual|irrevocable)\s+license|grant\s+of\s+license|license\s+to\s+use)[^.\n]+(?:\.[^.\n]+)?/i,
      risk_level: 'LOW',
      risk_score: 3,
      explanation: 'Grants permission to use software, content, or services under specified terms.',
      why_it_matters: 'Defines what you are allowed to do with the licensed product.',
      suggestion: 'Verify that the license scope matches your expected operational needs.'
    },
    {
      category: 'non_compete',
      regex: /(?:non-?compete|covenant\s+not\s+to\s+compete|shall\s+not\s+(?:directly\s+or\s+indirectly\s+)?engage\s+in\s+any\s+competitive|restrictive\s+covenant)[^.\n]+(?:\.[^.\n]+)?/i,
      risk_level: 'CRITICAL',
      risk_score: 9,
      explanation: 'Restricts working with competitors or operating in the same industry after the contract ends.',
      why_it_matters: 'Can severely prevent you from earning income or taking new clients.',
      suggestion: 'Narrow the geographic scope and duration (e.g. max 6 months), or negotiate its removal.'
    }
  ];

  const results: RawClauseDTO[] = [];
  const paragraphs = text.split(/\n\s*\n/).filter((p) => p.trim().length > 0);

  paragraphs.forEach((p, idx) => {
    for (const pat of patterns) {
      const match = p.match(pat.regex);
      if (match) {
        const clauseSnippet = match[0].trim();
        const start = text.indexOf(p);
        if (start !== -1) {
          const innerOffset = p.indexOf(clauseSnippet);
          const s = innerOffset !== -1 ? start + innerOffset : start;
          const e = s + (innerOffset !== -1 ? clauseSnippet.length : p.length);
          const exactText = text.slice(s, e);

          results.push({
            category: pat.category,
            clause_text: exactText,
            start_char: s,
            end_char: e,
            confidence: 0.95,
            chunk_id: `chunk_${idx}`,
            document_id: docId,
            page_number: 1,
            surrounding_context: text.slice(Math.max(0, s - 100), Math.min(text.length, e + 100)),
            model_version: 'clauseguard-pattern-v1',
            extraction_status: 'extracted',
            risk_level: pat.risk_level,
            risk_score: pat.risk_score,
            risk_reasoning: pat.why_it_matters,
            plain_english_explanation: pat.explanation,
            suggested_modification: pat.suggestion
          });
        }
        break;
      }
    }
  });

  return results;
}

export class MockClauseGuardApiClient implements ClauseGuardApiClient {
  private documents: Map<string, AnalysisResult> = new Map();
  private statuses: Map<string, 'pending' | 'processing' | 'completed' | 'failed'> = new Map();

  constructor() {
    const sample = getMockAnalysisResult();
    this.documents.set(sample.document_id, sample);
    this.statuses.set(sample.document_id, 'completed');
  }

  async uploadDocument(file: File): Promise<UploadResponse> {
    const docId = `doc_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`;

    // Read the text from file if possible, or fallback to sample
    let fileText = SAMPLE_CONTRACT_TEXT;
    let isRealTextUpload = false;
    try {
      if (file.type.includes('text') || file.name.endsWith('.txt')) {
        fileText = await file.text();
        isRealTextUpload = true;
      }
    } catch {
      fileText = SAMPLE_CONTRACT_TEXT;
    }

    let clauses: RawClauseDTO[];
    if (isRealTextUpload) {
      const dynamicClauses = extractDynamicClausesFromText(fileText, docId);
      clauses = dynamicClauses.length > 0
        ? dynamicClauses
        : getMockSampleClauses().map((c) => ({ ...c, document_id: docId }));
    } else {
      clauses = getMockSampleClauses().map((c) => ({
        ...c,
        document_id: docId
      }));
    }

    const result = adaptAnalysisResult({
      document_id: docId,
      filename: file.name,
      document_text: fileText,
      document_char_length: fileText.length,
      total_chunks: Math.ceil(fileText.length / 1000),
      total_extracted_clauses: clauses.length,
      pipeline_version: 'clauseguard-mock-pipeline-v1',
      clauses
    });

    this.documents.set(docId, result);
    this.statuses.set(docId, 'processing');

    return {
      document_id: docId,
      filename: file.name,
      status: 'uploaded',
      message: 'Document successfully uploaded and queued for ML processing.'
    };
  }

  async startAnalysis(documentId: string): Promise<AnalysisStatusResponse> {
    this.statuses.set(documentId, 'processing');
    return {
      document_id: documentId,
      status: 'processing',
      progress: 30,
      message: 'Running RoBERTa multi-label classification and 9-channel BIO span extraction...'
    };
  }

  async getAnalysisStatus(documentId: string): Promise<AnalysisStatusResponse> {
    // Transition from processing to completed after check
    const current = this.statuses.get(documentId) || 'completed';
    if (current === 'processing') {
      this.statuses.set(documentId, 'completed');
      return {
        document_id: documentId,
        status: 'completed',
        progress: 100,
        message: 'Analysis complete.'
      };
    }
    return {
      document_id: documentId,
      status: current,
      progress: current === 'completed' ? 100 : 50,
      message: current === 'completed' ? 'Analysis complete.' : 'Extracting contractual spans...'
    };
  }

  async getResults(documentId: string): Promise<AnalysisResult> {
    const doc = this.documents.get(documentId);
    if (doc) return doc;
    // Fallback to sample
    return getMockAnalysisResult(documentId);
  }

  async checkHealth(): Promise<HealthCheckResponse> {
    return {
      status: 'healthy',
      version: 'mock-1.0.0',
      model_loaded: true
    };
  }

  async getDocumentHistory(): Promise<DocumentHistoryItem[]> {
    // Stubbed mock history
    const sample = getMockAnalysisResult();
    return [
      {
        document_id: sample.document_id,
        filename: sample.filename,
        uploaded_at: '2026-11-12T16:00:00Z',
        total_clauses: sample.clauses.length,
        critical_count: sample.clauses.filter((c) => c.risk_level === 'CRITICAL').length,
        high_count: sample.clauses.filter((c) => c.risk_level === 'HIGH').length,
        medium_count: sample.clauses.filter((c) => c.risk_level === 'MEDIUM').length,
        low_count: sample.clauses.filter((c) => c.risk_level === 'LOW').length,
        needs_review_count: sample.clauses.filter((c) => c.risk_level === 'NEEDS_REVIEW').length
      }
    ];
  }
}

