import type { ClauseCategory } from '../types/analysis';

export interface CategoryMetadata {
  key: ClauseCategory;
  name: string;
  shortDescription: string;
  longDescription: string;
  iconName: string;
  typicalRisks: string[];
}

export const CLAUSE_CATEGORIES: Record<ClauseCategory, CategoryMetadata> = {
  liability_cap: {
    key: 'liability_cap',
    name: 'Liability Cap',
    shortDescription: 'Limits the monetary liability of one or both parties.',
    longDescription:
      'Defines the maximum financial damages a party can recover in a dispute, often capping recovery to fees paid or $0, leaving parties vulnerable to consequential damages.',
    iconName: 'ShieldAlert',
    typicalRisks: [
      'Nominal recovery caps ($0 or $100)',
      'Unilateral vendor liability disclaimers',
      'Exclusion of direct or consequential damages without carveouts'
    ]
  },
  termination_for_convenience: {
    key: 'termination_for_convenience',
    name: 'Termination for Convenience',
    shortDescription: 'Allows termination at will without proving cause or breach.',
    longDescription:
      'Grants discretionary rights to walk away from the contract without cause, which can lead to unexpected service cutoffs, loss of investments, or punitive exit fees.',
    iconName: 'FileX',
    typicalRisks: [
      'Unilateral immediate termination without notice',
      'Early termination penalty fees for customer',
      'Asymmetric notice periods'
    ]
  },
  anti_assignment: {
    key: 'anti_assignment',
    name: 'Anti-Assignment',
    shortDescription: 'Restricts transferring contract rights or business assets.',
    longDescription:
      'Regulates whether a party can assign rights or obligations, frequently blocking mergers, acquisitions, restructuring, or subcontracting without written consent.',
    iconName: 'ArrowLeftRight',
    typicalRisks: [
      'Absolute prohibition blocking mergers & acquisitions',
      'Unilateral assignment rights favoring one party',
      'Unreasonable counterparty withholding of consent'
    ]
  },
  renewal_term: {
    key: 'renewal_term',
    name: 'Renewal Term & Auto-Renew',
    shortDescription: 'Extends the contract duration and billing cycles automatically.',
    longDescription:
      'Governs how and when an agreement renews. Often incorporates automatic renewal traps requiring advance cancellation notice within an unreasonably narrow window.',
    iconName: 'RefreshCw',
    typicalRisks: [
      'Perpetual auto-renewal without notice reminder',
      'Excessive advance notice requirement (60–90+ days)',
      'Price escalation clauses upon automatic renewal'
    ]
  },
  governing_law: {
    key: 'governing_law',
    name: 'Governing Law & Jurisdiction',
    shortDescription: 'Establishes substantive law and judicial forum for disputes.',
    longDescription:
      'Specifies which jurisdiction’s laws apply and where lawsuits must be brought, potentially imposing extreme travel, litigation, and dispute resolution costs.',
    iconName: 'Scale',
    typicalRisks: [
      'Distant or hostile foreign jurisdiction',
      'Mandatory litigation venue far from user operations',
      'Waiver of local statutory consumer/business protections'
    ]
  },
  audit_rights: {
    key: 'audit_rights',
    name: 'Audit Rights',
    shortDescription: 'Grants counterparty inspection access to books, data, or facilities.',
    longDescription:
      'Compels access to internal records, software logs, servers, or physical facilities to verify compliance, potentially causing business disruption and confidentiality leaks.',
    iconName: 'SearchCheck',
    typicalRisks: [
      'Broad inspection access without advance notice',
      'Shifting full audit costs onto audited party for minor discrepancies',
      'Lack of reasonable confidentiality and security bounds'
    ]
  },
  ip_ownership_assignment: {
    key: 'ip_ownership_assignment',
    name: 'IP Ownership & Assignment',
    shortDescription: 'Transfers ownership of created intellectual property or code.',
    longDescription:
      'Transfers or claims ownership of work product, inventions, software, feedback, or trade secrets, risking forfeiture of pre-existing tools or derivative creations.',
    iconName: 'Cpu',
    typicalRisks: [
      'Overbroad assignment of background IP or pre-existing tooling',
      'Work-made-for-hire claims over independent work product',
      'Involuntary surrender of user feedback or derivatives'
    ]
  },
  license_grant: {
    key: 'license_grant',
    name: 'License Grant',
    shortDescription: 'Defines permissions, revocability, and bounds of asset usage.',
    longDescription:
      'Specifies what rights are granted to use software, content, or intellectual property. Critical concerns include exclusivity, perpetual rights, sublicensing, and revocability.',
    iconName: 'Key',
    typicalRisks: [
      'Irrevocable, perpetual, royalty-free license to customer content',
      'Severely restricted usage boundaries or seat counts',
      'Vendor right to suspend licenses unilaterally'
    ]
  },
  non_compete: {
    key: 'non_compete',
    name: 'Non-Compete Covenant',
    shortDescription: 'Restricts post-termination commercial and employment activities.',
    longDescription:
      'Prohibits engaging in competitive business, working for competitors, or soliciting customers post-agreement, potentially impairing future livelihood and commercial mobility.',
    iconName: 'Ban',
    typicalRisks: [
      'Excessive duration (e.g. 2+ years post-termination)',
      'Worldwide or unreasonably expansive geographic scope',
      'Broad industry definitions preventing legitimate freelance work'
    ]
  }
};

export const ALL_CATEGORY_KEYS: ClauseCategory[] = [
  'liability_cap',
  'termination_for_convenience',
  'anti_assignment',
  'renewal_term',
  'governing_law',
  'audit_rights',
  'ip_ownership_assignment',
  'license_grant',
  'non_compete'
];

export function getCategoryMeta(category: ClauseCategory): CategoryMetadata {
  return (
    CLAUSE_CATEGORIES[category] || {
      key: category,
      name: category.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase()),
      shortDescription: 'Contractual clause category.',
      longDescription: 'Contractual provision extracted from legal text.',
      iconName: 'FileText',
      typicalRisks: []
    }
  );
}

