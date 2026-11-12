# ClauseGuard Stage 5: Rule-Based Contextual Risk Assessment & Explanation Engine

## Executive Summary
This document specifies the architectural design, risk evaluation methodologies, heuristic legal rules, and explanation framework for **ClauseGuard Stage 5: Risk Assessment & Explanation Engine**.

Stage 5 bridges raw machine learning extractions and user-facing contract intelligence. It consumes localized clause spans and document metadata produced by Stage 4, applies deterministic, legally grounded heuristic rules across all nine CUAD categories, and outputs structured risk assessments with actionable explanations.

---

## 1. Core Architectural Principles

### 1.1 Decoupling Detection Confidence from Legal Risk
A fundamental architectural tenet in ClauseGuard is the strict separation between:
1. **Model Detection Confidence ($C \in [0.0, 1.0]$):** The statistical confidence that the span represents a specific clause category (from Stage 3 classifier and Stage 4 span extractor).
2. **Legal Risk Level ($\text{RiskLevel} \in \{\text{Low}, \text{Medium}, \text{High}, \text{Needs Review}\}$):** The contractual adversity, commercial liability exposure, or operational severity of the agreed terms.

> [!CAUTION]
> **Anti-Pattern Prohibited:** ClauseGuard never multiplies classifier probability by risk severity score. A high-risk clause (e.g., unlimited liability disclaimer) detected with $0.60$ model confidence remains a **High Risk** legal provision that requires urgent user attention. Conversely, a standard 60-day mutual termination clause detected with $0.99$ confidence is **Low Risk**.

### 1.2 Deterministic Heuristic Rules Over Uninterpretable ML
Contractual risk analysis in commercial environments requires transparency, auditability, and deterministic behavior:
- Black-box classifiers frequently hallucinate legal implications or make non-reproducible judgments.
- ClauseGuard implements rule-based evaluators using precise regular expressions, syntactic boundary checks, and legal drafting benchmarks vetted against commercial contracting standards.
- Every assigned risk level is directly traceable to concrete textual phrases within the contract.

### 1.3 Preservation of Unextracted Detections
In legal document analysis, false negatives are far more dangerous than false positives. If the Stage 3 classifier detects a high-risk category (e.g., `liability_cap`) at the document or chunk level, but Stage 4 fails to localize exact token boundaries:
- The detection is **never** silently discarded.
- Stage 5 creates an assessment finding with `extraction_status: "not_extracted"`, `risk_level: "Needs Review"`, and `review_required: true`.
- The user is alerted that the document likely contains the clause, prompting manual inspection.

### 1.4 Legal Disclaimer and Scope Boundaries
ClauseGuard provides automated contract analysis for risk triaging and informational assistance only. The engine does not provide formal legal advice, does not establish an attorney-client relationship, and does not declare clauses legally void or unenforceable under statutory law.

---

## 2. Risk Hierarchy & Evaluation Taxonomy

ClauseGuard categorizes contractual provisions into four standardized risk levels:

| Risk Level | Numeric Score Range | Definition | Operational Impact |
| :--- | :--- | :--- | :--- |
| **Low** | $1.0 - 3.5$ | Standard, balanced, or mutual commercial terms conforming to typical market practice. | Standard review; minimal negotiation needed. |
| **Medium** | $3.6 - 6.9$ | Modestly asymmetric or restrictive terms requiring careful consideration. | Target for minor negotiation or counter-proposal. |
| **High** | $7.0 - 10.0$ | Severe liability exposure, highly one-sided obligations, or oppressive restrictions. | Critical issue; redline and renegotiate prior to signing. |
| **Needs Review** | $5.0$ (Default) | Ambiguous drafting, unextracted detections, or incomplete textual fragments. | Manual review by a contract specialist required. |

---

## 3. Category-by-Category Legal Risk Rules

Each of the nine CUAD categories is evaluated by a dedicated analyzer subclassing `BaseClauseRuleAnalyzer`.

### 3.1 Limitation of Liability (`liability_cap`)
*Evaluates liability caps, disclaimers, and carve-outs.*

- **High Risk:**
  - Complete waiver or disclaimer of all liability ("shall not be liable for any damages whatsoever").
  - Nominal or de minimis dollar caps ($\le \$100$, e.g., "\$10" or "\$50").
  - Express carve-out excluding vendor breaches while customer liability remains unlimited.
- **Medium Risk:**
  - Fixed dollar cap between $\$100$ and $\$10,000$ in commercial enterprise contracts.
  - Multiplier caps (e.g., $2\times$ or $3\times$ annual fees).
  - Unilateral liability caps protecting only one party.
- **Low Risk:**
  - Standard industry benchmark: cap equal to fees paid in the prior twelve (12) months.
  - Mutual liability cap applying equally to both parties.
  - Standard mutual carve-outs for confidentiality breaches, IP indemnification, or gross negligence.

### 3.2 Termination for Convenience (`termination_for_convenience`)
*Evaluates cancellation rights, notice periods, and post-termination costs.*

- **High Risk:**
  - Immediate termination without cause or notice period under 15 days.
  - Unilateral convenience termination favoring only the counterparty.
  - Imposition of early termination penalties, forfeiture of prepaid fees, or exit fees.
- **Medium Risk:**
  - Notice period between 15 and 45 days.
  - Ambiguous termination notice mechanics.
- **Low Risk:**
  - Mutual right to terminate with at least 60 days advance written notice.
  - Explicit requirement for pro-rata refund of prepaid, unused service fees.

### 3.3 Anti-Assignment & Transfer Restrictions (`anti_assignment`)
*Evaluates restrictions on contract transfer and corporate transactions.*

- **High Risk:**
  - Absolute ban on assignment with zero exceptions ("any assignment shall be null and void").
  - Change-of-control triggered termination without opportunity to cure.
  - Unilateral assignment right favoring only the counterparty while binding the other.
- **Medium Risk:**
  - Assignment permitted only with consent, but lacking the standard reasonableness qualification ("which consent may be withheld in sole discretion").
- **Low Risk:**
  - Consent required but "shall not be unreasonably withheld, delayed, or conditioned".
  - Standard carve-outs permitting transfer to affiliates or in connection with a merger, acquisition, or sale of assets.

### 3.4 Renewal Term & Automatic Extension (`renewal_term`)
*Evaluates contract term duration, auto-renewal windows, and price escalations.*

- **High Risk:**
  - Evergreen auto-renewal with short cancellation notice windows ($< 30$ days) or onerous notice requirements.
  - Multi-year automatic lock-in terms (e.g., auto-renewing for 3 or 5-year successive periods).
  - Uncapped price escalation upon renewal ("at Vendor's then-current rates").
- **Medium Risk:**
  - Auto-renewal notice window between 30 and 59 days.
  - Annual renewal with capped price increases (e.g., $\le 5\%$ or CPI).
- **Low Risk:**
  - Contract requires mutual written consent or explicit opt-in for extension.
  - Ample notice window ($\ge 60$ days) prior to expiration to prevent unwanted renewal.

### 3.5 Governing Law & Jurisdiction (`governing_law`)
*Evaluates choice of law, dispute venue, and attorney fee provisions.*

- **High Risk:**
  - Selection of offshore tax havens, unfamiliar non-commercial foreign jurisdictions (e.g., Cayman Islands, Bermuda).
  - Unilateral attorney fee shifting (loser pays only if counterparty wins).
  - Inconvenient exclusive forum distant from primary business operations.
- **Medium Risk:**
  - Out-of-state domestic jurisdiction without exclusive venue clarity.
  - Unilateral jurisdiction selection.
- **Low Risk:**
  - Well-established neutral commercial law (Delaware, New York, California, England & Wales).
  - Mutual attorney fee shifting to the prevailing party.
  - Clear, accessible commercial venue.
- **Needs Review:**
  - Blank template placeholders (e.g., `[Insert State]`, `[Jurisdiction]`, `____`).

### 3.6 Audit Rights & Inspection (`audit_rights`)
*Evaluates financial, operational, and system inspection clauses.*

- **High Risk:**
  - Immediate, unannounced inspections without prior notice.
  - Broad, unrestricted inspection frequency ("at any time without limitation").
  - Shift of full audit costs to the audited party regardless of discrepancy findings.
- **Medium Risk:**
  - Short notice period (between 5 and 14 days).
  - Frequency exceeding twice per year.
- **Low Risk:**
  - Advance written notice of at least 30 days.
  - Audits restricted to normal business hours and no more than once per consecutive 12-month period.
  - Audit costs shifted only if an underpayment exceeding 5% is uncovered.

### 3.7 IP Ownership Assignment (`ip_ownership_assignment`)
*Evaluates title, patent/copyright transfers, work-for-hire, and background IP.*

- **High Risk:**
  - Broad assignment of all intellectual property, inventions, or improvements created during the term without carve-out for pre-existing IP.
  - Broad waiver of all moral rights.
  - Transfer of background IP, proprietary tools, or foundational code libraries.
- **Medium Risk:**
  - Ambiguous distinction between custom deliverables and underlying technology.
  - Customer ownership without license-back for reusable modules.
- **Low Risk:**
  - Explicit reservation of pre-existing background intellectual property.
  - Ownership transfer strictly confined to bespoke, custom deliverables paid for by customer.
  - Clear distinction between vendor core assets and customer data.

### 3.8 License Grant (`license_grant`)
*Evaluates scope, exclusivity, revocability, and commercial breadth.*

- **High Risk:**
  - Unilateral, perpetual, irrevocable license grant to customer data or core vendor IP for minimal consideration.
  - Exclusive license grant restricting licensor from licensing to other commercial parties.
  - Broad, royalty-free sublicensing rights to third parties without licensor approval.
- **Medium Risk:**
  - Ambiguous field-of-use restrictions.
  - Sublicensing rights without indemnity protections.
- **Low Risk:**
  - Standard non-exclusive, non-transferable, revocable commercial license.
  - Use restricted strictly to internal business purposes during the active subscription term.

### 3.9 Non-Compete & Restrictive Covenants (`non_compete`)
*Evaluates competitive restrictions, duration, geography, and non-solicitation.*

- **High Risk:**
  - Post-termination duration exceeding 12 months (e.g., 2, 3, or 5 years).
  - Worldwide or unbounded geographic scope.
  - Blanket restriction preventing ordinary commercial engagement in the entire industry.
- **Medium Risk:**
  - Duration between 6 and 12 months.
  - Moderate geographic scope restricted to defined operating markets.
- **Low Risk:**
  - Short duration ($\le 6$ months).
  - Restriction strictly limited to direct customer solicitation (non-solicitation) rather than blanket non-competition.

---

## 4. Explanation Framework & Grounding

Every finding generates a structured 4-part explanation:

```json
{
  "summary": "Plain-English synopsis of the provision's effect.",
  "why_it_matters": "Contextual commercial and legal risk explanation.",
  "supporting_evidence": "Verbatim contractual phrase justifying the rating.",
  "user_action": "Concrete, actionable negotiation redline recommendation."
}
```

### Grounding Requirements
1. **Verbatim Text Extraction:** The engine extracts exact character matches from `clause_text` (e.g., `"exceeding $10"`, `"laws of Delaware"`, `"sixty (60) days"`).
2. **Contextual Surrounding Snippet:** When full contract context is provided, surrounding text is supplied to give the reviewer contractual perspective.
3. **Actionable Redlines:** Every user action provides specific contracting language recommendations (e.g., *"Propose adding a mutual 12-month trailing fee liability cap"*).

---

## 5. Document-Level Risk Scoring & Aggregation

### 5.1 Document Overall Risk Level
The document-level risk category is derived hierarchically:

$$\text{Document Risk Level} = \begin{cases} 
\text{High} & \text{if } N_{\text{high}} > 0 \\ 
\text{Medium} & \text{if } N_{\text{high}} = 0 \text{ and } N_{\text{medium}} > 0 \\ 
\text{Needs Review} & \text{if } N_{\text{high}} = N_{\text{medium}} = 0 \text{ and } N_{\text{review}} > 0 \\ 
\text{Low} & \text{otherwise} 
\end{cases}$$

### 5.2 Document Composite Risk Score ($S \in [1.0, 10.0]$)
A continuous composite score is computed using weighted category risk contributions:
- If high-risk clauses exist, the document score is anchored in the upper bracket $[7.0, 10.0]$:
  $$S = \min\left(10.0, 7.0 + 0.6 \cdot (N_{\text{high}} - 1) + 0.2 \cdot N_{\text{medium}}\right)$$
- If medium-risk is the highest level:
  $$S = \min\left(6.9, 4.0 + 0.5 \cdot (N_{\text{medium}} - 1)\right)$$
- If low-risk only:
  $$S = \max\left(1.0, 2.0 - 0.2 \cdot N_{\text{low}}\right)$$
- If unextracted detections require review without explicit high risks:
  $$S = 5.0$$

---

## 6. Verification and Regression Suite

The Stage 5 implementation is validated by a 15-test automated suite in `ml/tests/test_stage5_risk_assessment.py`:
- 9 category-specific rule tests checking High, Medium, Low, and Needs Review conditions.
- Test 10: Insufficient evidence and fragment fallback.
- Test 11: Unextracted detection preservation (`extraction_status: "not_extracted"`).
- Test 12: Explanation 4-part structure and textual grounding.
- Test 13: Document-level aggregation and composite scoring.
- Test 14: Strict JSON serialization and Stage 6 schema compliance.
- Test 15: End-to-end integration consuming real `ClauseGuardPipeline` inference output.

**Test Results:**
- Stage 5 Test Suite: 15 / 15 PASSED (100%)
- Full Project Regression Suite: 46 / 46 PASSED (100%)
