# ClauseGuard Frontend (Stage 7)

ClauseGuard is an AI-powered legal document analysis application engineered to audit commercial contracts, SaaS Terms of Service (ToS), and independent contractor agreements. It identifies 9 canonical categories of contractual risks, evaluates their severity (`CRITICAL`, `HIGH`, `MEDIUM`, `LOW`, `NEEDS_REVIEW`), and explains them in plain English with negotiation counter-proposals.

This frontend client is built with **React 19**, **TypeScript** (strict mode), **Vite**, **Tailwind CSS**, **TanStack Query**, and **React Router**. It is designed with a **human-centred, document-first legal workspace aesthetic** for real freelancers, business owners, and reviewers without unnecessary AI hype or decorative visual clutter.

---

## 1. Quickstart & Local Development

### Prerequisites
- Node.js `^20.0.0` or `^22.0.0`
- npm `10+` or `11+`

### Installation & Startup
```bash
# Navigate to the frontend directory
cd frontend

# Install dependencies
npm install

# Start local Vite development server
npm run dev
```
The application will launch locally at `http://localhost:5173`.

### Build & Verification Commands
```bash
# Run full automated test suite (Vitest + React Testing Library)
npm test

# Run ESLint validation
npm run lint

# Typecheck and build production bundle
npm run build
```

---

## 2. Design System & Visual Principles

The application adopts a restrained, trustworthy legal productivity design:

- **Surface Palette:**
  - Main background: Warm white / light grey (`#F7F8FA`)
  - Document & content surfaces: Pure white (`#FFFFFF`)
  - Primary text: Dark charcoal (`#202938`)
  - Secondary text: Muted grey (`#687385`)
  - Primary accent: Navy / professional blue (`#315C8A`)
  - Borders: Subtle neutral grey (`#E4E7EC`)
- **Semantic Risk Palette:**
  - `CRITICAL`: Muted red (`#DC2626` / `bg-red-50 text-red-800 border-red-200`)
  - `HIGH`: Warm orange (`#EA580C` / `bg-orange-50 text-orange-800 border-orange-200`)
  - `MEDIUM`: Amber (`#D97706` / `bg-amber-50 text-amber-800 border-amber-200`)
  - `LOW`: Green (`#16A34A` / `bg-emerald-50 text-emerald-800 border-emerald-200`)
  - `NEEDS_REVIEW`: Neutral grey (`#64748B` / `bg-slate-100 text-slate-700 border-slate-200`)
- **Accessibility:**
  - Color is always paired with clear text labels and distinct geometric icons (`AlertOctagon`, `AlertTriangle`, `Info`, `CheckCircle`, `HelpCircle`).
  - WCAG AA contrast compliance across light and dark themes.
  - Interactive highlights and cards feature visible keyboard focus states and ARIA attributes.
- **Human-Centred Jargon Elimination:**
  - Internal ML terms (`chunk ID`, `pipeline version`, `BIO extraction`) are removed from user views.
  - Algorithmic certainty is explicitly labeled `"Detection confidence: XX%"` and visually isolated from legal risk severity.

---

## 3. Environment Variables & API Configuration

Configuration is managed via Vite environment variables. You can create a `.env` or `.env.local` file inside `frontend/`:

| Environment Variable | Default Value | Description |
|---|---|---|
| `VITE_API_BASE_URL` | `http://127.0.0.1:8000` | Base URL of the local FastAPI backend. |
| `VITE_USE_MOCK_API` | `true` | When `true` (default), uses the realistic in-memory Mock client (`src/lib/api/mock.ts`) with a ~14,000-character master agreement. Set to `false` to connect to the real FastAPI backend. |

### Switching Between Mock and Real Backend
The client uses a factory pattern in `src/lib/api/index.ts` implementing `ClauseGuardApiClient`:
1. **Mock Mode (Default):** Runs with zero backend dependencies. Perfect for UI development, testing, and demos. Loads a full commercial SaaS agreement with authentic examples across all 9 CUAD categories.
2. **Real Backend Mode:** Set `VITE_USE_MOCK_API=false` in `.env`. The frontend will issue standard `fetch` requests to `VITE_API_BASE_URL` (`POST /api/v1/documents/upload`, `POST /api/v1/analyze/{id}`, `GET /api/v1/results/{id}`, `GET /api/v1/health`).

---

## 4. Project Structure

```
frontend/src/
├── types/
│   └── analysis.ts          # Core domain models (ClauseCategory, RiskLevel, ReviewStatus, ExtractedClause, AnalysisResult)
├── lib/
│   ├── api/
│   │   ├── types.ts         # Backend DTOs & ClauseGuardApiClient interface
│   │   ├── client.ts        # Real fetch implementation (no axios)
│   │   ├── mock.ts          # Full ~14,000 character mock contract & sample clauses
│   │   ├── adapters.ts      # SINGLE-FILE translation isolating backend field renaming
│   │   └── index.ts         # Client singleton factory based on VITE_USE_MOCK_API
│   ├── highlight.ts         # Character-accurate text partitioner for overlapping/nested spans
│   ├── risk.ts              # Risk taxonomy, WCAG AA color palettes, sorting, count aggregation
│   ├── categories.ts        # The 9 canonical CUAD categories metadata, descriptions, icons
│   ├── export-pdf.ts        # Client-side printable PDF generator using jsPDF
│   └── utils.ts             # Tailwind class merging (cn), formatting, clipboard helpers
├── hooks/
│   ├── useAnalysis.ts       # TanStack Query hook with 2-second status polling
│   ├── useDocumentUpload.ts # Upload mutation & client-side file validation (<= 20MB)
│   ├── useBackendHealth.ts  # Readiness probe hook (/api/v1/health)
│   └── useTheme.ts          # Dark / light mode manager with localStorage persistence
├── components/
│   ├── layout/
│   │   ├── Navbar.tsx       # Minimal brand header, navigation, status dot, theme toggle
│   │   ├── Footer.tsx       # Mandatory educational disclaimer & local privacy notice
│   │   └── ErrorBoundary.tsx# React error boundary with fallback UI
│   ├── ui/
│   │   ├── RiskBadge.tsx    # Accessible badge (color-blind safe icon + label)
│   │   ├── CategoryBadge.tsx# Category icon + display name
│   │   └── ConfidenceBadge.tsx # Visually isolated detection certainty (NOT risk severity)
│   ├── upload/
│   │   ├── FileDropzone.tsx # Document drop area for PDF, DOCX, TXT with upload button
│   │   ├── UploadProgress.tsx # Minimal progress bar & retry handling
│   │   ├── CategoryShowcase.tsx # Clean reference grid of the 9 categories
│   │   └── SampleContractButton.tsx # Subtle secondary action to load sample agreement
│   ├── viewer/
│   │   ├── DocumentViewer.tsx # High-focus paper-like viewer with exact character highlights
│   │   ├── HighlightSegmentView.tsx # Interactive mark element with keyboard focus
│   │   ├── ViewerToolbar.tsx# Text search, font resizing, prev/next clause navigation
│   │   └── HighlightLegend.tsx # Color-blind safe inline legend
│   ├── risk/
│   │   ├── SummaryHeader.tsx# Compact document summary bar & PDF export trigger
│   │   ├── RiskCardList.tsx # Clean, divided vertical list with review checklist progress
│   │   ├── RiskCard.tsx     # Card with two-way sync to viewer highlight & status toggle
│   │   ├── RiskFilters.tsx  # Unobtrusive dropdown filters (Category, Risk level, Text search)
│   │   └── ExplanationDrawer.tsx # 5-section human-readable clause explanation drawer
│   └── history/
│       └── HistoryTable.tsx # Minimal audit history table (stubbed for Stage 6 persistence)
├── pages/
│   ├── UploadPage.tsx       # Route /
│   ├── ResultsPage.tsx      # Route /documents/:documentId (65/35 split workspace / stacked mobile)
│   └── HistoryPage.tsx      # Route /history
└── test/
    ├── highlight.test.ts    # Mathematical string preservation & overlap tests
    ├── risk.test.ts         # Risk hierarchy & multi-criteria filter tests
    ├── upload.test.tsx      # Dropzone & file validation component tests
    └── viewer-sync.test.tsx # Two-way synchronization tests (card <-> viewer)
```

---

## 5. Key Architectural & Product Invariants

1. **Confidence is NOT Risk:**
   - Model confidence (`confidence`) represents algorithmic detection certainty (0–100%).
   - Risk severity (`risk_level`) represents the substantive legal hazard (`CRITICAL`, `HIGH`, `MEDIUM`, `LOW`, `NEEDS_REVIEW`).
   - The UI never displays confidence as a "risk percentage". It is labeled `"Detection confidence: XX%"` and rendered in a visually separated neutral container.
2. **Detection Does Not Imply Hazard:**
   - A detected clause can be routine commercial standard (`LOW` risk). Low-risk clauses are rendered with green checkmarks, avoiding warning or alert aesthetics.
3. **NEEDS_REVIEW is Handled Honestly:**
   - When an unattached schedule or ambiguous wording is encountered, the badge displays a neutral `"Needs Review"` label. It is never treated as safe.
4. **Exact Character-Offset Preservation:**
   - Highlighting relies on `documentText.slice(start_char, end_char)` without trimming, lowercasing, or regex mutation.
   - `src/lib/highlight.ts` partitions the document text into contiguous intervals based on the union of all boundary points.
   - **Invariant Tested & Proven:** `segments.map(s => s.text).join('') === documentText`.
5. **Contract Review Checklist:**
   - Each clause tracks review status: `Not reviewed`, `Reviewed`, `Needs attention`. Users can cycle status directly on cards or in the details panel.
6. **Educational Disclaimer & Privacy:**
   - Persistent footer on every view: *"ClauseGuard is an educational document-analysis tool. It does not provide legal advice or replace consultation with a qualified legal professional."*
   - Documents are analyzed locally without sending data to external cloud APIs or analytics.

---

## 6. Backend Contract & Stage 6 Assumptions

The frontend connects to the following Stage 6 REST endpoints:
* `POST /api/v1/documents/upload`: Multipart file upload (`PDF`, `DOCX`, `TXT`). Returns `{ document_id: string, filename: string, status: string }`.
* `POST /api/v1/analyze/{document_id}`: Triggers the Stages 3–5 ML processing pipeline.
* `GET /api/v1/results/{document_id}`: Returns the full analysis payload matching the Stage 5 JSON schema.
* `GET /api/v1/health`: Readiness probe returning `{ status: "healthy" }`.
* `GET /api/v1/documents`: (Planned for Stage 6) Retrieves previous document audit history.
