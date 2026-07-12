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
- **Phase 2 — Full ML Pipeline Visibility:** see §10 below for the full spec.
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

---

## 10. Phase 2 — Full ML Pipeline Visibility

**Goal:** When a user uploads a `smartctl -j` JSON scan, they should see the complete
breakdown of how all 4 ML models contribute to the final HIR verdict — not just a
single number. Every model's individual score, intermediate data (reconstruction error,
bottleneck features, cluster assignment), reliability metrics, and feature-level
explanations should be visible. The user should understand *why* the system says what
it says.

### 10.1 The 4 Models Forming the HIR

The system fuses 4 independent ML models into a single Health Index Rating (HIR):

| Impl | Model | Type | Weight (backend) | Key metric | What it does |
|---|---|---|---|---|---|
| **Impl 0** | Sklearn Random Forest | Supervised | 0.20 | `failure_probability` | 19 SMART features + manufacturer → fail/healthy classification + HIR risk score |
| **Impl 2** | TF Bottleneck Classifier | Supervised (2-stage) | 0.50 | `failure_probability` | AE encoder (19→8 dim) → supervised classifier on bottleneck features. **Best performer: ROC-AUC 0.929, recall 89.1%** |
| **Impl 1** | TF Anomaly Autoencoder | Unsupervised | 0.10 | `anomaly_score` | Trained on 292k healthy rows only. Reconstruction error → normalized 0-1 anomaly score. Threshold at p99. |
| **Impl C** | UMAP + HDBSCAN | Unsupervised | 0.20 | `cluster_score` | Clusters on 8-dim bottleneck → assigns disk to a cluster with empirical failure rate. 18 clusters, 13.7% outliers. |

**Data flow for a single prediction:**
```
smartctl -j JSON
    │
    ├──→ pretvori_json_v_surovi_df()  →  raw DataFrame (19 SMART raw values + model + capacity)
    │                                    │
    │                                    ├──→ procesiraj_podatke()  →  feature engineering
    │                                    │    (any_critical_error, total_error_count, error_per_gb, jeSSD, ...)
    │                                    │
    │                                    ├──→ prepare_features()  →  19-column normalized feature vector
    │                                    │    │
    │                                    │    ├──→ [Impl 1] AE scaler → autoencoder → reconstruction error → anomaly_score
    │                                    │    │
    │                                    │    ├──→ [Impl 2] clf scaler → encoder → 8-dim bottleneck → classifier → failure_probability
    │                                    │    │                                                         │
    │                                    │    │                                                         └──→ [Impl C] HDBSCAN approximate_predict → cluster_id → cluster_score
    │                                    │    │
    │                                    │    └──→ [Impl 0] sklearn pipeline.analyze() → failure_probability + hir_risk_score
    │                                    │
    └──→ weighted average of 4 scores → disk_health_score → verdict (HEALTHY/AT_RISK/FAILURE)
```

### 10.2 What the Backend Currently Returns vs What It Computes

**Already returned by `/api/predict/combined`:**
```json
{
  "disk_health_score": 0.5,
  "verdict": "AT_RISK",
  "confidence": "medium",
  "model_scores": {
    "tf_classification": { "failure_probability": 0.45, "verdict": "AT_RISK", "weight": 0.5 },
    "tf_anomaly": { "anomaly_score": 0.15, "verdict": "HEALTHY", "weight": 0.1 },
    "clustering": { "cluster_score": 0.6, "cluster_label": "HIGH_RISK", "weight": 0.2 },
    "sklearn": { "failure_probability": 0.55, "hir_risk_score": 0.6, "verdict": "AT_RISK", "weight": 0.2 }
  },
  "consensus": { "models_predicting_failure": 3, "models_total": 4 }
}
```

**Computed but NOT returned (lost data):**

| Model | Field | Where in code | Why it matters |
|---|---|---|---|
| Anomaly AE | `reconstruction_error` | `_infer_anomaly()` L112 | Shows how far the disk deviates from "normal" — the raw signal before normalization |
| Anomaly AE | `threshold` | `_infer_anomaly()` L114 | The p99 cutoff — user can see if the disk is just barely over or way over |
| Bottleneck Clf | `bottleneck_features` (8-dim array) | `_infer_classification()` L159 | The compressed representation of this disk — shows where it sits in the model's learned space |
| Bottleneck Clf | `threshold`, `high_risk_threshold` | `_infer_classification()` L140-141 | The decision boundaries — user sees how close the disk is to the AT_RISK / FAILURE line |
| Clustering | `cluster_id` | `_infer_clustering()` L167 | Which of the 18 clusters this disk was assigned to |
| Clustering | `cluster_strength` | `_infer_clustering()` L168 | HDBSCAN confidence in the cluster assignment |
| Clustering | `cluster_failure_rate` | `_infer_clustering()` L175 | The empirical failure rate of this cluster (from training data) |
| Clustering | `is_outlier` | `_infer_clustering()` L172 | Whether HDBSCAN couldn't assign the disk to any cluster (66.9% failure rate for outliers!) |
| Sklearn RF | `models_output.classification_fail` | `disk_pipeline.py` L156 | The raw RF vote (fail/healthy) before probability |

### 10.3 Backend Enhancement Plan

**Goal:** Enrich the `/api/predict/combined` response to include all intermediate data.

Changes to `backend/main.py`:

1. **`_infer_anomaly()`** — already returns `reconstruction_error` and `threshold` in its
   dict, but `_infer_combined()` only extracts `anomaly_score` and `verdict`. Pass through
   the full dict.

2. **`_infer_classification()`** — already returns `bottleneck_features`, `threshold`, and
   `high_risk_threshold`. Pass these through to `model_scores.tf_classification`.

3. **`_infer_clustering()`** — already returns `cluster_id`, `is_outlier`, `cluster_strength`,
   `cluster_failure_rate`. Pass these through to `model_scores.clustering`.

4. **`_infer_sklearn()`** — already returns `models_output.classification_fail`. Pass through.

5. **Add a new `GET /api/models/metadata` endpoint** that returns all static model
   metadata (from the `*_metadata.json` files + feature importance) as a single JSON
   payload. This lets the frontend fetch it once on page load without bundling large
   JSON files at build time. Alternatively, keep using `import.meta.glob` for the
   metadata and skip this endpoint — **decision: use `import.meta.glob` for static
   metadata (simpler, no backend change needed), but enhance the combined response
   with the missing dynamic fields.**

**Enhanced response shape:**
```json
{
  "disk_health_score": 0.5,
  "verdict": "AT_RISK",
  "confidence": "medium",
  "model_scores": {
    "tf_classification": {
      "failure_probability": 0.45,
      "verdict": "AT_RISK",
      "weight": 0.5,
      "threshold": 0.2825,
      "high_risk_threshold": 0.65,
      "bottleneck_features": [0.12, -0.34, 0.56, 0.78, -0.23, 0.45, 0.67, -0.89]
    },
    "tf_anomaly": {
      "anomaly_score": 0.15,
      "verdict": "HEALTHY",
      "weight": 0.1,
      "reconstruction_error": 0.00321,
      "threshold": 0.00408,
      "is_anomaly": false
    },
    "clustering": {
      "cluster_score": 0.6,
      "cluster_label": "HIGH_RISK",
      "weight": 0.2,
      "cluster_id": 3,
      "is_outlier": false,
      "cluster_strength": 0.85,
      "cluster_failure_rate": 0.994,
      "cluster_total_samples": 167,
      "cluster_failure_samples": 166
    },
    "sklearn": {
      "failure_probability": 0.55,
      "hir_risk_score": 0.6,
      "verdict": "AT_RISK",
      "weight": 0.2,
      "classification_fail": true
    }
  },
  "consensus": {
    "models_predicting_failure": 3,
    "models_total": 4
  }
}
```

### 10.4 Static Model Metadata (bundled at build time)

All metadata is already in the repo. The frontend will bundle it via `import.meta.glob`:

| Source file | Data | Frontend use |
|---|---|---|
| `srcML/tensorflow_classification/bottleneck_metadata.json` | ROC-AUC 0.929, PR-AUC 0.934, failure recall 89.1%, F1 88.7%, bottleneck dim 8, threshold 0.283, training rows 6179/1324/1325, epochs 68/100 | Model Performance Panel — shows classifier reliability |
| `srcML/tensorflow_anomaly/tf_metadata.json` | ROC-AUC 0.901, PR-AUC 0.600, failure recall 44.7%, F1 55.5%, threshold 0.00408, p999 0.0322, training 292k healthy rows, epochs 37/60 | Model Performance Panel — shows AE reliability |
| `srcML/tensorflow_clustering/hdbscan_metadata.json` | 18 clusters, 1207 outliers (13.67%), per-cluster risk_label/risk_score/failure_rate/total_samples/failure_samples, UMAP params, HDBSCAN params | Model Performance Panel + Cluster Detail view |
| `srcML/tensorflow_classification/clf_ae_metadata.json` | Stage 1 AE metadata: ROC-AUC 0.839, threshold 0.0177, training 292k healthy, epochs 60/60 | Model Performance Panel — shows the encoder's standalone quality |
| `DiskJson/bottleneck_sweep_results.json` | 6 bottleneck dims tested (4,6,7,8,10,12), per-dim ROC-AUC/PR-AUC/recall/precision/F1, recommended dim=8 | Optional: show why dim=8 was chosen |
| `srcML/sklearn/feature_importance.csv` (gitignored — hardcode in JS) | Top features: error_per_gb (16.8%), any_critical_error (14.8%), total_error_count (10.3%), smart_9 (7.3%), smart_197 (7.1%), smart_5 (5.6%) | Explainability Panel — which SMART attributes drive predictions |
| `Graphs/*.png` (9 images) | Training curves, ROC, UMAP+HDBSCAN, confusion matrices, HIR formula | Graph Gallery |

**Note on `feature_importance.csv`:** Both `.gitignore` and `.dockerignore` exclude
`*.csv`. The file has only 20 rows — hardcode the data as a JS constant in
`src/api/modelMetadata.js` to avoid build issues.

### 10.5 Frontend Component Specs

#### 10.5.1 Explainability Panel (replaces `.widget-shap` stub, `span 4, row span 2`)

**Goal:** Show which SMART features drive the prediction + this drive's actual values.

**Sections (top to bottom in the tall card):**

1. **Feature Importance bars** (always visible, static data):
   - Top 10 features, horizontal bars sorted by importance descending
   - Bar color: `--status-critical` for SMART 5/187/188/197/198, `--status-warning` for
     SMART 9 (age), `--accent-blue` for others
   - Each bar: feature label (left), importance % (right, `tabular-nums`)

2. **This Drive's Values** (shown only after a scan):
   - For each top-10 feature, show this drive's actual raw value
   - Map feature names to SMART IDs: `smart_5_raw` → attr id=5, etc.
   - Non-zero critical features (SMART 5/187/197/198) → row highlighted with
     `--status-critical-bg` + warning icon
   - Engineered features (`error_per_gb`, `any_critical_error`, `total_error_count`)
     computed client-side from the SMART attributes

3. **RF Verdict detail** (shown only after a scan):
   - `classification_fail: true/false` — the raw RF vote
   - `hir_risk_score` — the RF's own HIR score (0-100, clamped 5-97)

**Data flow:** Static from `modelMetadata.js`, dynamic from `result.model_scores.sklearn`
+ client-side parsed `smartctl JSON`.

#### 10.5.2 Model Performance Panel (replaces `.widget-trend` stub, `span 8`)

**Goal:** 4-column model comparison showing reliability metrics + this scan's results.

**Layout: 4 mini-cards side by side, each with:**

**Card header:** Model name + role badge + weight badge

**Card body — static reliability metrics (from metadata):**
- ROC-AUC (with progress bar 0-1)
- PR-AUC
- Failure Recall (with progress bar)
- Failure Precision
- Failure F1
- Training info (rows, epochs converged/requested)

**Card body — this scan's results (from API response, highlighted section):**
- This scan's score (large, colored by status)
- This scan's verdict (pill badge)
- Model-specific extra data:
  - **RF:** HIR risk score, classification_fail vote
  - **Bottleneck Clf:** failure probability vs threshold (visual gauge showing how close
    to the AT_RISK / FAILURE boundary), 8-dim bottleneck features as a mini sparkline/bar
  - **Anomaly AE:** reconstruction error vs threshold (visual gauge showing if over/under
    the p99 cutoff), is_anomaly flag
  - **HDBSCAN:** cluster_id, cluster_label, cluster_failure_rate, is_outlier flag,
    cluster_strength (confidence in assignment)

**Card footer:** Tooltip icon with plain-language explanation of the model's role

**Data flow:** Static from `modelMetadata.js`, dynamic from enhanced
`/api/predict/combined` response.

#### 10.5.3 Model Consensus Enhancement (existing `.widget-health`)

**Goal:** Add reliability indicators + richer per-model data.

**Changes to `HealthWidget.jsx`:**
- Below each mini-donut, add a tiny reliability bar (3px, `--radius-pill`) colored by ROC-AUC:
  - ≥0.90 → `--status-healthy`, ≥0.80 → `--status-warning`, <0.80 → `--status-critical`,
    N/A → `--status-neutral`
- Below the donuts, add a compact "consensus breakdown" line:
  `3/4 models flag failure · weights: Clf 0.50 · HDBSCAN 0.20 · RF 0.20 · AE 0.10`

**Data flow:** Static from `modelMetadata.js`, dynamic from existing `result` props.

#### 10.5.4 Graph Gallery (optional, below main grid or modal)

**Goal:** Show all 9 pre-generated training graphs.

- Triggered from a "View Training Graphs" button in the Model Performance Panel
- Modal overlay or collapsible section below the main grid
- Responsive grid of thumbnails with lightbox/zoom on click
- Graphs bundled via `import.meta.glob('../../Graphs/*.png', { query: '?url', eager: true })`

**Graph label mapping:**

| Filename | Label |
|---|---|
| `nn_classification.png` | Bottleneck Classifier — Training Curves |
| `nn_autoencoder.png` | Autoencoder — Training & Reconstruction Error |
| `classification.png` | Random Forest — Classification Results |
| `regression.png` | Random Forest — HIR Regression Results |
| `clustering.png` | Clustering — UMAP + HDBSCAN Visualization |
| `umap_hdbscan.png` | UMAP Projection with HDBSCAN Clusters |
| `bottleneck_kmeans_clusters.png` | Bottleneck Space — K-means Clusters |
| `kmeans_elbow.png` | K-means Elbow Plot |
| `hir_formula.png` | HIR Formula Diagram |

### 10.6 Implementation Steps

**Step 1 — Backend: Enhance `/api/predict/combined` response**
- In `backend/main.py`, pass through the missing fields from each `_infer_*()` function
  into `model_scores` (see §10.3 for the exact enhanced shape)
- No new endpoints needed — just enrich the existing response
- Verify with `curl -X POST http://localhost:8000/api/predict/combined -F "file=@DiskJson/used_with_errors.json"`

**Step 2 — Frontend: Create `src/api/modelMetadata.js`**
- Hardcode `FEATURE_IMPORTANCE` array (20 rows, since CSV is gitignored)
- Use `import.meta.glob` for the 4 metadata JSON files + `bottleneck_sweep_results.json`
- Export a structured `MODEL_METADATA` object with all static metrics
- Export `featureLabel()` and `featureToSmartId()` helper functions
- Export `GRAPH_LABELS` mapping for the graph gallery

**Step 3 — Frontend: Rework `ShapWidget.jsx` → Explainability Panel**
- Keep `.widget-shap` class for grid compatibility
- Feature importance bars (static) + this drive's values (dynamic)
- Empty state: "Upload a scan to see feature-level contributions"
- Loading state: skeleton bars

**Step 4 — Frontend: Rework `TrendWidget.jsx` → Model Performance Panel**
- Keep `.widget-trend` class for grid compatibility
- 4-column mini-card grid with static metrics + dynamic per-scan data
- Visual gauges for threshold comparisons (AE reconstruction error, Clf failure prob)
- Bottleneck features sparkline
- Cluster detail with outlier warning
- "View Training Graphs" button

**Step 5 — Frontend: Enhance `HealthWidget.jsx`**
- Add reliability bars under each mini-donut
- Add consensus breakdown line

**Step 6 — Frontend: Create `GraphGallery.jsx`**
- Modal or collapsible section
- Bundle `Graphs/*.png` via `import.meta.glob`
- Lightbox/zoom on click

**Step 7 — Frontend: Wire `Dashboard.jsx`**
- Pass `result` (enhanced) + `smartData` to the new Explainability Panel
- Pass `result` to the Model Performance Panel
- Pass `result` + static metadata to enhanced HealthWidget
- Add GraphGallery trigger

**Step 8 — CSS: Add all new styles to `index.css`**
- Feature importance bars, model comparison grid, reliability indicators
- Threshold gauges, bottleneck sparkline, cluster detail
- Graph gallery modal/thumbnails
- All using §3.5 design tokens

**Step 9 — Docker: Update build context**
- Update `.dockerignore` to allow `Graphs/` (currently excluded)
- Update `frontend/Dockerfile` to `COPY Graphs/ ./Graphs/` before build
- Verify `srcML/tensorflow_*/*.json` files are accessible to `import.meta.glob`
  (they're currently in the build context since `.dockerignore` only excludes `*.pkl`
  and `*.keras` globally with selective re-includes — the JSON files should pass through)

**Step 10 — Verify end-to-end**
- `docker compose up --build`
- Upload each sample JSON from `DiskJson/`
- Verify all 4 model scores + intermediate data appear
- Verify feature importance + drive values appear
- Verify graph gallery works
- Verify `tabular-nums` on all numeric values

### 10.7 Libraries

- **No new npm dependencies.** Feature importance bars, threshold gauges, and
  bottleneck sparklines all use plain CSS divs (same pattern as the Phase 1 SMART
  attributes panel). `recharts` can be added later if interactive charts are needed.

### 10.8 Definition of Done (Phase 2)

- [ ] Backend `/api/predict/combined` returns all intermediate data: `reconstruction_error`,
      `threshold`, `bottleneck_features`, `cluster_id`, `cluster_strength`,
      `cluster_failure_rate`, `is_outlier`, `classification_fail`.
- [ ] `ShapWidget` slot (`.widget-shap`) shows feature importance bars from RF, sorted
      descending, with critical SMART attributes highlighted.
- [ ] After a scan, the explainability panel shows this drive's actual values for each
      important feature, with non-zero critical features flagged.
- [ ] `TrendWidget` slot (`.widget-trend`) shows a 4-column model comparison with
      ROC-AUC, PR-AUC, recall, precision, F1, training info, plus this scan's per-model
      scores/verdicts/intermediate data.
- [ ] Model Performance Panel shows model-specific extras: AE reconstruction error vs
      threshold gauge, Clf failure prob vs threshold gauge, bottleneck features sparkline,
      HDBSCAN cluster detail with outlier warning.
- [ ] Model Consensus (`.widget-health`) shows reliability bars under each mini-donut
      + consensus breakdown line.
- [ ] (Optional) Graph gallery shows all 9 `Graphs/*.png` with human-readable labels.
- [ ] All new CSS uses §3.5 design tokens.
- [ ] Every numeric value uses `font-variant-numeric: tabular-nums`.
- [ ] `docker compose up` works end-to-end with all new static assets + enriched API.
