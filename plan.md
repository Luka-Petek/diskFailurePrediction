# Frontend Dashboard Plan — Disk Failure Prediction

> Scope: `frontend/` (React + Vite). Goal: a simple, clean, effective dashboard showing a
> final failure % indicator plus supporting disk-health information, with room to grow
> into performance/explainability graphs later. This plan intentionally keeps the ML
> pipeline (`srcML/`, `backend/`) as the source of truth — the UI only visualizes it.

---

## 1. ML System Recap (context — do not contradict this in the UI)

Four independent models are trained in `srcML/` and fused into one score:

| Model | Location | README weight |
|---|---|---|
| Sklearn Random Forest | `srcML/sklearn/disk_pipeline.py` | 0.30 |
| TF Bottleneck Classifier | `srcML/tensorflow_classification/` | 0.40 |
| TF Autoencoder (anomaly) | `srcML/tensorflow_anomaly/` | 0.20 |
| HDBSCAN clustering | `srcML/tensorflow_clustering/` | 0.10 |

The **FastAPI backend** (`backend/main.py`) exposes:
- `POST /api/predict/anomaly` — Impl 1 result (`anomaly`, `anomaly_score`, `verdict`).
- `POST /api/predict/classification` — Impl 2 result (`failure_probability`, `verdict`, `bottleneck_features`).
- `POST /api/predict/clustering` — HDBSCAN cluster result (`cluster_label`, `cluster_score`).
- `POST /api/predict/sklearn` — Impl 0 result (`hir_risk_score`, `failure_probability`, `verdict`).
- `POST /api/predict/combined` — **primary endpoint for the dashboard.** Returns:
  ```json
  {
    "disk_health_score": 0.0-1.0,
    "verdict": "HEALTHY" | "AT_RISK" | "FAILURE",
    "confidence": "none" | "low" | "medium" | "high",
    "model_scores": { "<model>": { ...score, "verdict", "weight" } },
    "consensus": { "models_predicting_failure": n, "models_total": n }
  }
  ```
- `POST /api/analyze-smart-json` — legacy alias, sklearn-only.

All endpoints accept a raw `smartctl -A -i -j` JSON file upload (multipart `file` field).
Example shape in `DiskJson/disk_data_sda.json`: `model_name`, `user_capacity.bytes`,
`ata_smart_attributes.table[]` (each item has `id`, `name`, `value`, `raw.value`).

### ⚠️ Known inconsistency (resolve before treating the % as final/authoritative)
`srcML/hir_final.py` documents the README's HIR formula: RMS combination, weights
**0.30/0.40/0.20/0.10**, clamped to `[3, 97]`, thresholds **40/75** → HEALTHY/WARNING/CRITICAL.

`backend/main.py`'s actual `/api/predict/combined` (what the frontend will call) uses a
**different** formula: plain weighted average, weights **0.50/0.10/0.20/0.20**, thresholds
**0.40/0.70** → HEALTHY/AT_RISK/FAILURE.

These two "final scores" will disagree on the same input. Recommendation: pick ONE
canonical formula (ideally make `backend/main.py` call into `srcML/hir_final.py`'s logic,
or update the README/weights to match the backend) before/while wiring the frontend, so the
headline number is trustworthy. This is a backend/ML fix, not a UI task — the UI will simply
display whatever `/api/predict/combined` returns, correctly labeled with its own verdict enum.

---

## 2. Current Frontend State (audit)

- Stack: React 19 + Vite, plain CSS, no chart library, no HTTP client — `package.json`
  only lists `react`/`react-dom`.
- `src/Dashboard.jsx` renders a static grid of widgets with **zero state/data-fetching**.
- All components (`StatusWidget`, `HealthWidget`, `ShapWidget`, `TrendWidget`,
  `BarChartWidget`, `LogsWidget`) contain **hardcoded fake numbers**.
- Bug: `src/components/Navbar.jsx` uses a block-bodied arrow function with no `return`
  statement — it currently renders nothing.
- `src/css/index.css` has a solid dark-SaaS token system (`--bg-main`, `--accent-*`,
  `--radius`) and a working 12-column CSS grid — **keep this foundation**.
- Docker: frontend served via nginx on `:3000`, backend on `:8000`, CORS is already
  open (`allow_origins=["*"]`) — direct `fetch` from the browser to `localhost:8000`
  works in both dev and docker-compose without a proxy.

---

## 3. UX / Design Goals

Guiding principles, distilled from dashboard-design best practice (Tufte's data-ink ratio,
F-pattern scanning, bento-grid hierarchy — see refs at the bottom of §3.9):

- **One primary metric.** A single, unambiguous failure-risk % + verdict badge anchors the
  **top-left** of the grid — the zone with the highest visual weight in an F-pattern scan —
  not four competing donuts fighting for attention.
- **Maximize data-ink, minimize chrome.** No 3D effects, no gratuitous shadows/gradients.
  Every pixel of color/border must carry meaning. Flat cards, subtle borders — refine the
  existing demo's flat aesthetic, don't add decoration on top of it.
- **Color is semantic, never decorative.** Green/amber/red are reserved exclusively for
  health status (healthy/warning/critical). Blue is reserved for neutral UI actions
  (buttons, links, focus rings). Never reuse a status color for something that isn't a
  status.
- **Trustworthy, not a black box.** Show the per-model breakdown and consensus next to the
  headline number so the ensemble's reasoning is visible, not hidden.
- **Actionable.** Surface the actual SMART attributes driving the score (SMART 5, 187, 188,
  197 per the README's top predictors), visually flagged, not just listed.
- **Room to grow.** Reserve clearly-labeled, visually distinct (dashed border) slots for
  Phase 2 graphs so the layout doesn't shift later.
- **Density over whitespace, but not clutter.** Dashboard users want data at a glance —
  use the spacing scale below consistently rather than ad-hoc margins.

---

## 3.5 Design System — Tokens (extend `src/css/index.css`, don't replace it)

The existing tokens (`--bg-main: #0c0d12`, `--bg-card: #14151a`, `--border-color: #21232c`,
`--text-main: #f0f0f0`, `--text-muted: #8b8f9e`, `Inter` font, `--radius: 12px`) are a solid
foundation — **keep every one of them.** Add the following, additively, at the top of
`src/css/index.css`:

```css
:root {
  /* --- keep all existing tokens above, add these --- */

  /* Elevated surface — hover/active state, dropdowns, skeleton highlight */
  --bg-elevated: #1a1c23;
  --border-color-strong: #2c2f3a;

  /* Semantic status — used ONLY for health/verdict, never decoratively */
  --status-healthy: #22c55e;
  --status-healthy-bg: rgba(34, 197, 94, 0.12);
  --status-warning: #f59e0b;
  --status-warning-bg: rgba(245, 158, 11, 0.12);
  --status-critical: #ef4444;
  --status-critical-bg: rgba(239, 68, 68, 0.12);
  --status-neutral: #64748b;
  --status-neutral-bg: rgba(100, 116, 139, 0.12);

  /* Brand accents — decorative only (CTA buttons, focus ring, nav highlight) */
  --accent-blue-hover: #2f74e0;

  /* Spacing scale — 4px base unit, use for ALL margin/padding/gap */
  --space-1: 4px;  --space-2: 8px;  --space-3: 12px; --space-4: 16px;
  --space-5: 20px; --space-6: 24px; --space-8: 32px; --space-10: 40px;
  --space-12: 48px; --space-16: 64px;

  /* Radius scale */
  --radius-sm: 8px;
  --radius-lg: 16px;
  --radius-pill: 999px;

  /* Elevation — flat by default; a hint of lift ONLY on hover of interactive cards */
  --shadow-hover: 0 8px 24px rgba(0, 0, 0, 0.35);

  /* Motion */
  --transition-fast: 120ms ease-out;
  --transition-base: 200ms ease-out;
}
```

**Note:** `--accent-pink` stays in the palette as a decorative brand accent only (e.g. an
active-tab underline) — it must **not** be reused for "danger/critical" anymore now that
real semantic status tokens exist (the current demo's `danger-circle` misuses pink for this;
switch it to `--status-critical` when repurposing).

**Typography scale** (`Inter`, already imported via Google Fonts in `index.css`):

| Use | Size / line-height | Weight | Notes |
|---|---|---|---|
| Hero % number | 64px / 1 | 800 | `font-variant-numeric: tabular-nums` |
| Section/page title | 24px / 1.3 | 700 | rarely needed, navbar logo already covers this |
| Card title (existing `.card-title`) | 13px / 1.4, uppercase, 0.4px letter-spacing | 600 | keep as-is |
| Secondary metric value (drive info, SMART raw values) | 18px / 1.2 | 700 | `tabular-nums` |
| Body / labels | 14px / 1.5 | 500 | default |
| Meta text (timestamps, units, confidence) | 12px / 1.4 | 500 | `--text-muted` |

Apply `font-variant-numeric: tabular-nums;` to **every** numeric value on the dashboard
(hero %, sub-scores, SMART raw values, power-on hours) so digits align vertically as they
change between scans — small detail, big perceived-polish difference.

**Icons:** reuse the existing sprite at `frontend/public/icons.svg` first. If it's missing
icons needed for the new components (upload, check-circle, alert-triangle, x-circle,
hard-drive, clock, cpu), add `lucide-react` — a few KB per icon, tree-shakeable, and matches
the existing minimal aesthetic far better than an icon font.

---

## 3.6 Layout Blueprint — Preserve the Existing Composition

**Decision:** the current demo's composition is already good — a donut hero top-left, a
tall right-hand card, two wide chart cards, and a compact list card. That's already the
F-pattern/hero-top-left convention best-practice research recommends. Phase 1 therefore
**keeps every existing `grid-column`/`grid-row` span in `index.css` exactly as-is** and only
re-skins/re-wires what's *inside* each slot with real data. No new full-width row is added —
the upload control moves **into the Navbar** (see §3.7) so the six-card body stays untouched.

Existing spans (unchanged) mapped to their Phase 1 real-data role:

| Existing class | `grid-column` / `grid-row` | Was (fake demo) | Becomes (real data) |
|---|---|---|---|
| `.widget-status` | `span 4` / row 1 | "ML Verdict Today" donut | **Hero Verdict Card** — already the hero slot, top-left |
| `.widget-health` | `span 4` / row 1 | 3 fake subsystem donuts | **Model Consensus** — 4 mini-donuts (RF/Clf/AE/HDBSCAN) + drive-identity line |
| `.widget-shap` | `span 4`, `row span 2` | fake SHAP donuts | **Explainability stub** — "Coming soon" (Phase 2), tall slot suits it |
| `.widget-trend` | `span 8` | fake anomaly chart | **Trend stub** — "Coming soon" (Phase 2, needs scan history) |
| `.widget-bar` | `span 8` | fake SMART bars | **SMART Attributes panel** — real bars from `ata_smart_attributes.table[]` |
| `.widget-logs` | `span 4` | fake critical logs | **Recent Scans** — real, session-local list, ships as real data (no stub needed) |

This is the **default**, not a hard mandate — it's the known-good starting point precisely
because the existing composition already tested well. If, while implementing, a genuinely
cleaner structure turns up for a given slot (e.g. 4 consensus donuts cramped at `span 4`,
or a table layout that reads better than bars for SMART attributes), change it. Don't
force real data into a shape that fights it just to avoid touching CSS.

Existing responsive behavior in `index.css` is untouched too — no new breakpoints needed
since no new slot shapes are introduced.

---

## 3.7 Component Specs (mapped onto the existing slots above)

**Hero Verdict Card** (`.widget-status`, unchanged span — top-left, row 1)
- Radial gauge: CSS `conic-gradient` ring — same technique already used for
  `.donut`/`.status-donut` — keep the existing 120px diameter, just drive the progress
  color from the current status token (`--status-healthy|warning|critical`) instead of the
  hardcoded `--accent-yellow`.
- Center: the existing `.donut-text h2` slot, bumped to 28px/800, `tabular-nums`.
- The old `<span>Risk</span>` label becomes a **status pill** — `border-radius:
  var(--radius-pill)`, `padding: 4px 12px`, `background: var(--status-*-bg)`, `color:
  var(--status-*)`, uppercase 11px/600 — HEALTHY / AT RISK / FAILURE, mirroring the
  backend's `verdict`.
- The existing `.status-legend` column (currently two fake `legend-item`s) becomes:
  confidence ("Confidence: high") + disk identity (model + short serial).
- **Empty state** (no scan yet): ring at 0% in `--status-neutral`, "—" instead of a number,
  legend reads "Upload a SMART report to begin".

**Model Consensus** (`.widget-health`, unchanged span, row 1)
- Add one compact identity row above the existing `.mini-donuts` (the card's
  `flex-direction: column` + `margin-top: auto` on `.mini-donuts` already reserves that top
  space): disk model + capacity, 12px muted.
- `.mini-donuts` grows from 3 to 4 items (RF / Bottleneck Clf / Anomaly AE / HDBSCAN); each
  `.md-circle` uses `conic-gradient` colored by that model's own status token, `.md-inner`
  shows its sub-score, `.subsystem-label` shows the model name + weight ("RF · w0.30").

**Explainability stub** (`.widget-shap`, unchanged tall span)
- Replace the fake donuts/matrix grid with the shared stub pattern: centered icon (32px),
  "Coming soon" (14px/700), one line describing what Phase 2 will show (per-feature SHAP
  contributions). `border: 1px dashed var(--border-color-strong)`, `background:
  transparent` — visually distinct from live-data cards. No hover effect.

**Trend stub** (`.widget-trend`, unchanged span)
- Same stub pattern: "Coming soon — anomaly trend over time (needs scan history)".

**SMART Attributes panel** (`.widget-bar`, unchanged span)
- Replace the fake `.bars` with real bars/rows: one per `ata_smart_attributes.table[]`
  entry, label = attribute name, value = `raw.value` (right-aligned, bold, `tabular-nums`).
- The known top predictors (SMART 5, 187, 188, 197) get a 3px left border strip in the
  matching status color instead of the flat `bg-pink` used for every bar today.

**Recent Scans** (`.widget-logs`, unchanged span) — real data, not a stub
- Reuse the existing `.log-item` row layout: left = disk model + short serial, right =
  that scan's risk %, colored via the status token.
- Backed by simple client-side session state (array of past `/api/predict/combined`
  results this session) — no backend storage needed, so it ships as real data in Phase 1.

**Navbar / Upload control** (`.navbar`, unchanged span)
- Keep the existing logo + nav-links exactly as styled.
- Replace the decorative `.avatar` circle with a primary **"Upload Scan"** button
  (`background: var(--accent-blue)`, `color: #fff`, `border-radius: var(--radius-sm)`,
  `padding: 8px 16px`) opening a small popover: drag-drop dropzone + a dropdown of bundled
  `DiskJson/*.json` samples + an "Analyze" button. Keeps the 6-card body untouched.

**Buttons / CTA (general)**
- Primary ("Analyze", "Upload Scan"): as above; hover → `var(--accent-blue-hover)`;
  `transition: var(--transition-fast)`.
- Secondary (sample picker, "See More"): transparent background, `1px solid
  var(--border-color)`, `color: var(--text-main)`; hover → `border-color:
  var(--border-color-strong)`.

---

## 3.8 States — loading / empty / error (design these, don't skip them)

- **Loading:** skeleton placeholders matching each card's real shape, shimmer animation:
  ```css
  .skeleton {
    background: linear-gradient(90deg, var(--bg-card) 25%, var(--bg-elevated) 37%, var(--bg-card) 63%);
    background-size: 400% 100%;
    animation: skeleton-shimmer 1.4s ease-in-out infinite;
  }
  @keyframes skeleton-shimmer { 0% { background-position: 100% 50%; } 100% { background-position: 0 50%; } }
  ```
- **Error:** a dismissible banner directly under the Navbar — `background:
  var(--status-critical-bg)`, `color: var(--status-critical)`, left icon + message (e.g.
  "Invalid SMART JSON" / "Backend unavailable") — never a full-page takeover.
- **Empty** (first load, no scan yet): Hero card shows the neutral placeholder from §3.7;
  Consensus, SMART panel, and Recent Scans show a single muted line ("No scan yet — upload
  a SMART report above") instead of rendering empty boxes.

---

## 3.9 Interaction & Accessibility

- Hover, only on genuinely interactive elements (upload zone, sample picker, "See More"):
  `transform: translateY(-2px); box-shadow: var(--shadow-hover); transition:
  var(--transition-base);`. Static info cards (drive info, SMART table) get **no** hover —
  they're not clickable, don't imply affordance that isn't there.
- `:focus-visible` on every interactive element: `outline: 2px solid var(--accent-blue);
  outline-offset: 2px;`.
- Wrap the Hero card's number in `aria-live="polite"` so screen readers announce the new
  result after an upload completes.
- Contrast check: `--text-muted` (#8b8f9e) on `--bg-card` (#14151a) ≈ 4.6:1, passes WCAG AA
  for normal text — keep this exact pairing, don't lighten `--text-muted` further or it
  will start failing contrast on smaller meta text.

*References used for §3–3.9: dashboard information-hierarchy and bento-grid patterns
(zone-based layout, F-pattern scanning, tile-sizing-by-priority), gauge/KPI-card best
practice (one metric per gauge, 3-zone semantic color, context via target/trend), and
dark-mode contrast/token conventions — synthesized into the concrete values above rather
than left abstract.*

---

## 4. Information Architecture (Phase 1)

Maps 1:1 onto the six existing card slots (see §3.6) — no new cards, no new rows:

1. **Upload control** — lives in the Navbar, not a new row: drag-drop a `smartctl -j`
   JSON file, or pick one of the bundled `DiskJson/*.json` samples.
2. **Hero verdict card** (`.widget-status` slot) — big %, verdict badge, confidence.
   Sourced from `disk_health_score` / `verdict` / `confidence`.
3. **Model consensus** (`.widget-health` slot) — 4 mini-donuts (RF, Bottleneck
   Classifier, Anomaly AE, HDBSCAN) with sub-score, weight, and vote — from
   `model_scores{}` — plus a compact drive-identity line (model + capacity).
4. **SMART attributes panel** (`.widget-bar` slot) — real bars from
   `ata_smart_attributes.table[]`, known top-predictor attributes visually highlighted.
5. **Recent scans** (`.widget-logs` slot) — real, session-local list of past uploads, no
   backend storage required.
6. **Insights stubs** (`.widget-shap` and `.widget-trend` slots) — explicit **"Coming
   soon"** placeholders (explainability, trend-over-time) instead of fake numbers — IA
   stays stable for Phase 2, no fabricated data.

---

## 5. Component / Code Plan

All changes below **reuse the existing six grid slots exactly as they are** (see §3.6) —
nothing is added or removed from the grid, only what renders inside each slot changes.

- **Fix:** `src/components/Navbar.jsx` — add the missing `return`; replace the decorative
  `.avatar` circle with an "Upload Scan" button opening a small popover (drag-drop
  dropzone + a dropdown of bundled `DiskJson/*.json` samples + "Analyze" button). This is
  the *only* structural addition — no new grid row.
- **New:** `src/api/client.js` — fetch wrapper for multipart POST to the backend, reading
  a `VITE_API_BASE_URL` env var (default `http://localhost:8000`).
- **New:** `src/hooks/useDiskAnalysis.js` (or a small context) — holds the selected file,
  loading/error state, a session-local scan-history array, and the latest
  `/api/predict/combined` result.
- **Update:** `src/Dashboard.jsx` — owns the upload flow via the hook above, passes the
  result down to widgets instead of each widget being static.
- **Rework `StatusWidget.jsx` → Hero Verdict card** (stays in its existing `.widget-status`
  slot) — implements the gauge/pill/confidence spec in §3.7.
- **Rework `HealthWidget.jsx` → Model Consensus** (stays in its existing `.widget-health`
  slot) — 4 mini-donuts from `model_scores{}` plus a drive-identity line, per §3.7.
- **Rework `BarChartWidget.jsx` → SMART Attributes panel** (stays in its existing
  `.widget-bar` slot) — real bars from the uploaded JSON, left-border risk flagging for
  SMART 5/187/188/197.
- **Rework `LogsWidget.jsx` → Recent Scans** (stays in its existing `.widget-logs` slot) —
  real, session-local list of past results, no backend storage needed.
- **Stub (no removal, no relocation):** `TrendWidget.jsx`, `ShapWidget.jsx` — apply the
  shared `.widget-stub` dashed-border style from §3.7 with a "Coming soon" state; their
  grid classes (`widget-trend`, `widget-shap`) stay untouched so Phase 2 slots in without
  any layout change.
- **New `src/components/Skeleton.jsx`** — reusable shimmer block per §3.8, used by every
  widget while `loading` is true.
- **CSS:** add the §3.5 tokens and §3.7 component-level classes to `src/css/index.css` —
  keep every existing token, and every existing `grid-column`/`grid-row` span, exactly as
  they are today (see §3.6).

---

## 6. Backend Touch Points (light, only what's needed for Phase 1)

- Frontend calls `POST /api/predict/combined` once per uploaded scan — it already
  returns everything Phase 1 needs (score, verdict, confidence, per-model breakdown).
- No new backend endpoint required for the SMART attributes panel — parsed entirely
  client-side from the uploaded `smartctl` JSON.
- Resolve the weight/threshold inconsistency noted in §1 before shipping — otherwise the
  dashboard's headline % won't match the README's documented HIR behavior.

---

## 7. Libraries

- **Phase 1:** none needed — CSS `conic-gradient` donuts (already used in the demo) are
  enough for the hero gauge and consensus mini-donuts.
- **Phase 2:** add `recharts` when real time-series/SHAP charts are built (lightweight,
  React-idiomatic, no D3 boilerplate).

---

## 8. Phased Roadmap

- **Phase 1 (this pass):** wire the real upload → analyze flow into the existing six card
  slots (Hero, Model Consensus + drive info, SMART attributes panel, Recent Scans); fix
  `Navbar` bug and add its Upload Scan control; stub Trend/SHAP widgets with "Coming soon"
  instead of fake data. No grid/composition changes.
- **Phase 2 (later):**
  - Performance graphs — start cheap by embedding the already-generated
    `Graphs/*.png` (ROC curves, autoencoder architecture, UMAP+HDBSCAN plot), then
    consider live-computed charts via `recharts`.
  - Explainability — a small new backend endpoint exposing per-feature contributions
    (raw material already exists: `srcML/sklearn/feature_importance.csv`,
    `srcML/sklearn/X_test_shap.csv`), rendered as a SHAP-style bar chart in place of
    the current `ShapWidget` stub.
- **Phase 3 (optional):** persistent scan history. Today the backend is fully
  stateless — every `/api/predict/*` call is a one-shot inference with no storage.
  "Recent scans" would need either a lightweight DB/JSON log on the backend or an
  explicit client-side-only (session/localStorage) history, scoped accordingly.

---

## 9. Definition of Done (Phase 1)

- [ ] `Navbar.jsx` renders correctly (bug fixed) and exposes the "Upload Scan" control.
- [ ] Uploading a `smartctl -j` JSON (or picking a `DiskJson/` sample) triggers a real
      call to `/api/predict/combined` and renders the response.
- [ ] Hero card (`.widget-status` slot) shows the real failure % (from
      `disk_health_score`), verdict, and confidence — no hardcoded numbers left.
- [ ] Model Consensus (`.widget-health` slot) shows real per-model scores/weights/votes
      from `model_scores`, plus the drive-identity line.
- [ ] SMART attributes panel (`.widget-bar` slot) shows real values from
      `ata_smart_attributes.table[]`.
- [ ] Recent Scans (`.widget-logs` slot) shows real session-local scan history, not fake
      log entries.
- [ ] Trend/SHAP widgets use the shared `.widget-stub` dashed style with a "Coming soon"
      placeholder, not fake data.
- [ ] Loading and error states are handled (bad JSON, backend unavailable, model not
      loaded → `503` from `_require()` in `backend/main.py`) using the skeleton/banner
      patterns in §3.8.
- [ ] All new CSS uses the §3.5 tokens (spacing scale, status colors, radius scale) —
      no ad-hoc hex codes or magic-number pixel values introduced.
- [ ] The existing six-card composition (see §3.6) was used as the starting point; any
      deviation from it (span/structure changes) was a deliberate improvement made during
      implementation, not an accidental drift.
- [ ] Every numeric value uses `font-variant-numeric: tabular-nums`.
- [ ] Status colors (`--status-healthy|warning|critical`) are used **only** for
      health/verdict semantics — not reused decoratively anywhere else on the page.
