---
description: Verify all citations, claims, and terminology in the academic paper before submission
---

# Paper Citation & Content Verification Workflow

Use this workflow before any paper submission or major revision to ensure
all citations, factual claims, and terminology are accurate.

## 1. Verify Each Citation Against Its Source

For every `\cite{...}` in the paper:

- Open the original source (PDF, DOI link, or arXiv page) — do NOT trust
  secondhand summaries or AI-generated descriptions.
- Confirm: authors, year, venue, and the specific claim being attributed.
- Check that the citation key in `references.bib` matches the actual paper.
- If you cannot access the source, flag it with `% TODO: VERIFY` in the LaTeX.

## 2. Verify Factual Claims About Prior Work

For every sentence that describes what another paper does:

- Does the paper actually use the model/method you claim?
  - e.g., "SVM" vs "LSTM" vs "autoencoder" — check the method section.
- Does the paper actually report the metric you cite?
  - e.g., "0.95 F-measure" — find the exact number in the results.
- Are you oversimplifying? If a paper uses multiple models, mention all
  relevant ones, not just one.

## 3. Verify Dataset Claims

- Dataset name, year, size — confirm against the official source.
- Class imbalance ratios — recompute or verify from the data.
- HDD vs SSD proportions — confirm from dataset documentation or code.
- Number of failures, number of records — verify from raw data or stats.

## 4. Verify Model Descriptions Match the Codebase

- Model names in paper must match model names in `srcML/` code.
- Architecture details (layers, bottleneck dimension, clustering algorithm)
  must match the actual implementation.
- Performance metrics (ROC-AUC, recall, F1) must match `DiskJson/` results.

## 5. Copyright & Permissions Check

- Confirm the paper template (acmart.cls) is the correct version for the
  target venue (IS2026 / SiKDD 2026).
- Check whether any figures or tables reproduced from other sources require
  permission — if so, add a credit line and retain evidence of permission.
- Verify the `\setcopyright` setting matches the venue's requirements.
- If using excerpts from copyrighted datasets (e.g., Backblaze), confirm the
  dataset license allows academic use and cite it appropriately.

## 6. Terminology Consistency

- Run a grep for banned terms: "fusion", "fusing", "fused" — replace with
  "hybrid model", "aggregation", "aggregated".
- "Aggregated Health Index (AHI)" — always capitalized.
- "Failure Assessment" (not "Failure Prediction") in the title.
- Check all three LaTeX files (main.tex, paper.tex, paper_full.tex) for
  consistency.

## 7. Cross-File Consistency

- Title must match across main.tex, paper.tex, paper_full.tex.
- Abstract claims must not contradict introduction or results.
- Section labels (`\label{sec:...}`) must match `\ref{sec:...}` references.
- Keywords should be consistent across files.

## 8. Final Compilation Check

- Run: `pdflatex main && biber main && pdflatex main && pdflatex main`
- Check for warnings about undefined references or citations.
- Verify all `\ref{}` and `\cite{}` resolve correctly.
- Check that no TODO comments remain in the final submission version.
