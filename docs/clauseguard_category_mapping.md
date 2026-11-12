# ClauseGuard | Category Mapping & MVP Selection Specification

**Project:** ClauseGuard (Legal Document Analyzer)  
**Document:** Category Mapping and Selection Strategy  
**Source Dataset:** Contract Understanding Atticus Dataset (CUAD) v1  
**Execution Date:** November 12, 2026  
**Status:** Approved for Stage 2 Preprocessing  

---

## 1. Overview & Selection Methodology

The goal of ClauseGuard is to analyze legal contracts, Terms of Service (ToS), and privacy policies, identifying high-risk, one-sided, or burdensome provisions.

While CUAD provides 41 categories annotated by legal professionals, training an effective multi-label transformer model requires focusing on a robust, high-signal subset for the initial Minimum Viable Product (MVP). Training on all 41 categories simultaneously would introduce severe class imbalance failure modes, as 16 of the 41 categories appear in fewer than 15% of contracts.

### Selection Criteria for MVP Categories:
1. **Empirical Dataset Density:** Must possess sufficient training volume ($\ge 100$ positive contracts and $\ge 200$ annotated spans across the 510 contracts).
2. **Substantive Risk Relevance:** Must directly inform consumer, freelancer, or enterprise risk (e.g., liability exposure, unilateral termination, restrictive covenants), rather than purely administrative metadata (e.g., party names, document title).
3. **Clause-Level Annotation Quality:** Must represent full-sentence or multi-sentence legal provisions (median word length $\ge 25$ words) suitable for paragraph classification and document highlighting.
4. **Clean Semantic Boundaries:** Must avoid extreme conflation or ambiguous definitions that confuse transformer attention heads.

---

## 2. Product Concepts vs. CUAD Ground Truth

The project context identified 10 high-priority product concepts. Below is the objective audit comparing each product concept against actual CUAD annotations:

| Product-Level Concept | CUAD Status | Actual CUAD Candidate Labels | Suitability for MVP |
| :--- | :---: | :--- | :---: |
| **1. Liability Limitations** | **Supported** | `Cap on Liability` (53.9%), `Uncapped Liability` (21.8%) | **Selected (MVP #1)** |
| **2. Termination** | **Partial** | `Termination for Convenience` (35.9%), `Post-Termination Services` (35.7%) | **Selected (MVP #2)** |
| **3. Assignment / 3rd-Party** | **Supported** | `Anti-Assignment` (73.3%), `Change of Control` (23.7%) | **Selected (MVP #3)** |
| **4. Automatic Renewal** | **Supported** | `Renewal Term` (34.5%), `Notice Period to Terminate Renewal` (21.8%) | **Selected (MVP #4)** |
| **5. Governing Law** | **Supported** | `Governing Law` (85.7%) | **Selected (MVP #5)** |
| **6. User Obligations / Audit** | **Supported** | `Audit Rights` (42.0%) | **Selected (MVP #6)** |
| **7. Intellectual Property** | **Supported** | `IP Ownership Assignment` (24.3%), `Joint IP Ownership` (9.0%) | **Selected (MVP #7)** |
| **8. Licensing Rights** | **Supported** | `License Grant` (50.0%), `Non-Transferable License` (27.1%) | **Selected (MVP #8)** |
| **9. Restrictive Covenants** | **Supported** | `Non-Compete` (23.3%), `Exclusivity` (35.3%) | **Selected (MVP #9)** |
| **10. Indemnification** | **MISSING** | *None* (No general indemnification category exists in CUAD) | **Gapped (Requires External Data)** |
| **11. Confidentiality / NDA** | **MISSING** | *None* (No confidentiality category exists in CUAD) | **Gapped (Requires External Data)** |
| **12. Dispute / Arbitration** | **MISSING** | *None* (Only `Governing Law`; no arbitration / jury waiver) | **Gapped (Requires External Data)** |

---

## 3. Recommended 9 MVP Categories

We recommend a focused **9-Category MVP**. Every recommended category has been validated against the empirical metrics computed in `docs/dataset_statistics.json`.

```
ClauseGuard MVP Taxonomy
├── Financial & Legal Exposure
│   ├── liability_cap                   (CUAD: Cap on Liability)
│   └── governing_law                   (CUAD: Governing Law)
├── Operational & Contract Freedom
│   ├── termination_for_convenience     (CUAD: Termination for Convenience)
│   ├── renewal_term                    (CUAD: Renewal Term)
│   └── anti_assignment                 (CUAD: Anti-Assignment)
├── Intellectual Property & Licensing
│   ├── ip_ownership_assignment         (CUAD: IP Ownership Assignment)
│   └── license_grant                   (CUAD: License Grant)
└── Operational Restrictions & Burden
    ├── non_compete                     (CUAD: Non-Compete)
    └── audit_rights                    (CUAD: Audit Rights)
```

---

### Category Profiles

#### 1. Limitation of Liability (`liability_cap`)
- **Original CUAD Label:** `Cap on Liability`
- **Assigned Risk Level:** **HIGH**
- **Dataset Coverage:** 275 contracts (53.9% coverage) | 672 spans (Train: 535, Test: 137) | Avg 2.44 spans/pos doc
- **Span Length:** Median: 49 words (306 chars) | Max: 299 words
- **Relevance to Users:** Essential risk clause. Flags provisions where the service provider caps liability to \$0, a nominal fee, or fees paid in the last 12 months, shielding them from substantial damages while leaving the user unprotected.
- **Annotation Suitability:** High. Annotated as complete sentences or comprehensive clauses. 96%+ contained within single paragraphs.
- **Mapping Limitations & Nuance:** Frequently co-occurs with `Uncapped Liability` (141 overlapping spans), where exceptions (e.g. willful misconduct, IP infringement) are carved out. For MVP, we target the liability cap language itself.

#### 2. Unilateral Termination (`termination_for_convenience`)
- **Original CUAD Label:** `Termination for Convenience`
- **Assigned Risk Level:** **HIGH**
- **Dataset Coverage:** 183 contracts (35.9% coverage) | 246 spans (Train: 196, Test: 50) | Avg 1.34 spans/pos doc
- **Span Length:** Median: 27 words (163 chars) | Max: 333 words
- **Relevance to Users:** Critical for ToS, vendor contracts, and employment. Indicates whether either party can terminate the agreement at will without demonstrating breach or cause.
- **Annotation Suitability:** Excellent. Clean, concise, full-sentence clauses (median 27 words).
- **Mapping Limitations & Nuance:** CUAD does **not** annotate termination for cause/breach (which is ubiquitous boilerplate in commercial contracts). The model will specifically flag *convenience* or discretionary termination rights.

#### 3. Anti-Assignment & Transfer Restrictions (`anti_assignment`)
- **Original CUAD Label:** `Anti-Assignment`
- **Assigned Risk Level:** **MEDIUM**
- **Dataset Coverage:** 374 contracts (73.3% coverage) | 654 spans (Train: 532, Test: 122) | Avg 1.75 spans/pos doc
- **Span Length:** Median: 35 words (218 chars) | Max: 375 words
- **Relevance to Users:** Restricts a user or business from selling, transferring, or assigning their account, business assets, or contractual rights without explicit counterparty consent.
- **Annotation Suitability:** Very high density. 73.3% of all contracts contain this clause.
- **Mapping Limitations & Nuance:** Shares 34 overlapping spans with `Change of Control`. Contracts frequently bundle assignment restrictions with corporate restructuring clauses.

#### 4. Automatic Renewal Terms (`renewal_term`)
- **Original CUAD Label:** `Renewal Term`
- **Assigned Risk Level:** **HIGH**
- **Dataset Coverage:** 176 contracts (34.5% coverage) | 210 spans (Train: 167, Test: 43) | Avg 1.19 spans/pos doc
- **Span Length:** Median: 44 words (270 chars) | Max: 461 words
- **Relevance to Users:** High consumer and SaaS relevance. Flags trap clauses that automatically lock users into recurring billing cycles or multi-year extensions.
- **Annotation Suitability:** Good. Median length of 44 words provides rich semantic context for auto-renewal phrasing.
- **Mapping Limitations & Nuance:** In CUAD, `Renewal Term` defines the extension mechanism, while `Notice Period to Terminate Renewal` (122 spans) specifies the required opt-out deadline (e.g. 30 or 60 days). They overlap heavily (105 pairs). In MVP, `renewal_term` encompasses auto-extension conditions.

#### 5. Governing Law & Jurisdiction (`governing_law`)
- **Original CUAD Label:** `Governing Law`
- **Assigned Risk Level:** **LOW to MEDIUM**
- **Dataset Coverage:** 437 contracts (85.7% coverage) | 464 spans (Train: 374, Test: 90) | Avg 1.06 spans/pos doc
- **Span Length:** Median: 29 words (178 chars) | Max: 141 words
- **Relevance to Users:** Informs users of the legal venue and governing state/country laws (e.g. Delaware, New York, England). Unfavorable foreign jurisdictions significantly increase dispute costs.
- **Annotation Suitability:** Highly uniform, clean, single-sentence annotations across 85.7% of all documents.
- **Mapping Limitations & Nuance:** CUAD labels specify the governing jurisdiction, but do **not** indicate whether disputes must go to arbitration. It is strictly a choice-of-law / venue identifier.

#### 6. Audit & Inspection Rights (`audit_rights`)
- **Original CUAD Label:** `Audit Rights`
- **Assigned Risk Level:** **MEDIUM to HIGH**
- **Dataset Coverage:** 214 contracts (42.0% coverage) | 643 spans (Train: 516, Test: 127) | Avg 3.00 spans/pos doc
- **Span Length:** Median: 40 words (253 chars) | Max: 288 words
- **Relevance to Users:** Imposes burdensome requirements permitting the counterparty to enter premises, inspect server logs, or review financial books.
- **Annotation Suitability:** Robust representation (643 spans across 214 contracts). High signal-to-noise ratio.
- **Mapping Limitations & Nuance:** Commercial contracts often specify inspection notice (e.g., 5 business days) and who pays for the audit. The model learns the overarching inspection covenant.

#### 7. IP Ownership & Assignment (`ip_ownership_assignment`)
- **Original CUAD Label:** `IP Ownership Assignment`
- **Assigned Risk Level:** **HIGH**
- **Dataset Coverage:** 124 contracts (24.3% coverage) | 318 spans (Train: 266, Test: 52) | Avg 2.56 spans/pos doc
- **Span Length:** Median: 52 words (337 chars) | Max: 300 words
- **Relevance to Users:** Paramount for developers, contractors, creators, and SaaS users. Flags clauses where work product, user data, or intellectual property is forcibly transferred or assigned to the company.
- **Annotation Suitability:** Clear, substantive legal paragraphs with 52 median words.
- **Mapping Limitations & Nuance:** Should not be conflated with `Joint IP Ownership` (only 46 contracts, 115 spans). MVP focuses strictly on outright assignment and ownership transfer.

#### 8. License Scope & Restrictions (`license_grant`)
- **Original CUAD Label:** `License Grant`
- **Assigned Risk Level:** **MEDIUM**
- **Dataset Coverage:** 255 contracts (50.0% coverage) | 777 spans (Train: 647, Test: 130) | Avg 3.05 spans/pos doc
- **Span Length:** Median: 53 words (356 chars) | Max: 394 words
- **Relevance to Users:** Defines what rights the user receives (or grants to the platform). Distinguishes whether software or content usage is revocable, limited, or conditioned.
- **Annotation Suitability:** Extensive training data (777 spans, highest non-metadata clause count).
- **Mapping Limitations & Nuance:** Broad category that frequently overlaps with `Non-Transferable License` (216 overlaps) and `Irrevocable or Perpetual License` (155 overlaps).

#### 9. Restrictive Covenants / Non-Compete (`non_compete`)
- **Original CUAD Label:** `Non-Compete`
- **Assigned Risk Level:** **HIGH**
- **Dataset Coverage:** 119 contracts (23.3% coverage) | 259 spans (Train: 206, Test: 53) | Avg 2.18 spans/pos doc
- **Span Length:** Median: 48 words (309 chars) | Max: 479 words
- **Relevance to Users:** High legal risk. Barring an individual or company from engaging in competing business, working for competitors, or operating within certain geographic zones.
- **Annotation Suitability:** Dense paragraphs (median 48 words, up to 479 words) with explicit restrictive terminology.
- **Mapping Limitations & Nuance:** CUAD also contains `No-Solicit of Customers` (34 contracts) and `No-Solicit of Employees` (59 contracts). However, customer and employee non-solicitation are too sparse for independent MVP training. MVP focuses strictly on commercial non-compete clauses.

---

## 4. Evaluated but Deferred Categories

The following table documents why the remaining 32 categories in CUAD were excluded from the Phase 1 MVP:

| Category | Spans | Positive Docs (%) | Rationale for Deferral |
| :--- | :---: | :---: | :--- |
| **Document Name** | 521 | 100.0% | Administrative metadata; not a risk clause. |
| **Parties** | 2,554 | 99.8% | Named entity recognition (NER) task, not clause risk classification. |
| **Agreement Date** | 476 | 92.2% | Metadata entity; rule-based regex is more appropriate than transformer heads. |
| **Effective Date** | 447 | 76.5% | Metadata entity; rule-based regex is more appropriate. |
| **Expiration Date** | 467 | 81.0% | Subsumed partly by Term & Renewal; high overlap with Agreement Date. |
| **Insurance** | 560 | 32.5% | Specific to commercial enterprise vendor deals; secondary priority for ToS/consumer. |
| **Post-Termination Services** | 450 | 35.7% | Specialized transition services in M&A/enterprise; lower consumer relevance. |
| **Revenue/Profit Sharing** | 418 | 32.5% | Commercial monetization term; distinct from core consumer legal risk. |
| **Minimum Commitment** | 424 | 32.4% | B2B purchasing quota; deferred to commercial enterprise add-on module. |
| **Exclusivity** | 410 | 35.3% | High overlap with License Grant; deferred to Phase 2 to prevent label confusion. |
| **Rofr/Rofo/Rofn** | 367 | 16.7% | Specialized commercial rights of first refusal/offer; low ToS relevance. |
| **Non-Transferable License** | 298 | 27.1% | Heavily nested inside `License Grant` (216 overlaps); redundant for initial MVP. |
| **Change of Control** | 253 | 23.7% | Subsumed partly in `Anti-Assignment`; candidate for Phase 2 expansion. |
| **Warranty Duration** | 176 | 14.7% | Moderate sparsity; hybrid entity/clause format. |
| **Covenant Not to Sue** | 173 | 19.6% | Specialized patent/litigation clause; moderate sparsity (100 docs). |
| **Volume Restriction** | 171 | 16.1% | Commercial manufacturing/distributor quota; low ToS relevance. |
| **Uncapped Liability** | 167 | 21.8% | Strongly co-occurs with `Cap on Liability`; deferred to fine-grained carveout detection. |
| **Irrevocable / Perpetual Lic.** | 165 | 13.7% | Nested within `License Grant`; severe sparsity (<15% coverage). |
| **Competitive Restr. Except.** | 125 | 14.9% | Exception subclause; severe sparsity (76 docs). |
| **Notice to Terminate Renewal** | 122 | 21.8% | Highly correlated with `Renewal Term`; merged into renewal concept. |
| **Liquidated Damages** | 121 | 12.0% | Low coverage (61 docs); candidate for later financial penalty module. |
| **Affiliate License-Licensee** | 115 | 11.6% | Low coverage (59 docs); nested licensing term. |
| **Joint IP Ownership** | 115 | 9.0% | Extreme sparsity (46 docs); high risk of overfitting. |
| **No-Solicit of Employees** | 91 | 11.6% | Low coverage (59 docs); insufficient volume for standalone classification. |
| **Affiliate License-Licensor** | 69 | 4.5% | Extreme sparsity (23 docs). |
| **Source Code Escrow** | 66 | 2.5% | Extreme sparsity (13 docs). |
| **Non-Disparagement** | 65 | 7.5% | Extreme sparsity (38 docs); critical ToS clause, but requires external data. |
| **No-Solicit of Customers** | 58 | 6.7% | Extreme sparsity (34 docs). |
| **Third Party Beneficiary** | 39 | 6.3% | Extreme sparsity (32 docs). |
| **Most Favored Nation** | 38 | 5.5% | Extreme sparsity (28 docs). |
| **Unlimited/All-You-Can-Eat** | 32 | 3.3% | Extreme sparsity (17 docs). |
| **Price Restrictions** | 27 | 2.9% | Extreme sparsity (15 docs). |

---

## 5. Critical Dataset Gaps & External Data Requirements

Our empirical investigation confirms that **three top-tier ClauseGuard product features cannot be supported using CUAD alone**. Attempting to force these features onto CUAD would compromise model integrity:

### Gap 1: Indemnification Obligations
- **The Need:** One of the greatest risks in software, contractor, and ToS agreements is one-sided indemnification (e.g. *"User agrees to defend, indemnify, and hold harmless the Company against all third-party claims..."*).
- **The CUAD Reality:** CUAD v1 contains **zero** categories for indemnification. While `Uncapped Liability` occasionally lists indemnification as an uncapped exception, it does not annotate the core indemnity obligation.
- **Remediation Strategy:** In Stage 3, augment ClauseGuard with the **LEDGAR dataset** (Tuggener et al., LREC 2020), which contains over 60,000 legal provisions from SEC filings, including thousands of dedicated `Indemnification` clauses.

### Gap 2: Mandatory Arbitration & Class Action Waivers
- **The Need:** In Terms of Service, privacy policies, and consumer contracts, mandatory binding arbitration and class action waivers are prime consumer hazards that extinguish court access.
- **The CUAD Reality:** CUAD annotates `Governing Law` (which state's substantive law applies), but contains **no annotation for arbitration, dispute resolution forums, or jury waivers**.
- **Remediation Strategy:** Ingest the **CLAUDETTE dataset** (Lippi et al., 2019), an expert-annotated corpus of unfair terms in online Terms of Service that explicitly labels arbitration and jurisdiction traps.

### Gap 3: Confidentiality & Non-Disclosure (NDA) Provisions
- **The Need:** Users need to detect confidentiality obligations, trade secret restrictions, and survival terms.
- **The CUAD Reality:** The Atticus Project explicitly omitted confidentiality because virtually all 510 contracts contained it, making it uninformative for their corporate transaction study.
- **Remediation Strategy:** Extract standard confidentiality clauses from the open-source **SEC EDGAR Form 8-K Exhibits (Item 1.01)** or the **LEDGAR corpus** under the `Confidentiality` provision category.

---

## 6. Summary of Category Strategy for Stage 2

For Stage 2 (Data Preprocessing and Pipeline Implementation), the data transformation pipeline will:
1. Target the **9 MVP categories** defined above.
2. Filter the 41 original CUAD labels down to these 9 targets.
3. Map every positive annotation span for these 9 categories to their corresponding paragraph chunks.
4. Construct a unified multi-hot binary label vector:
   $$\mathbf{y} = [y_1, y_2, \dots, y_9] \in \{0, 1\}^9$$
5. Retain exact document character offsets to enable document highlighting in the ClauseGuard web interface.
