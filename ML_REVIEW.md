# ML Engineering Review — DiskGuard

Review of the full project from an ML engineering perspective (all 4 implementations, shared preprocessing, HIR fusion, backend, Docker). Sorted by severity. Items marked with ⚠️ are the ones most likely to be challenged at a conference.

---

## What's done well

- **Reproducibility**: seeds set everywhere (`np`/`random`/`tf`), artifacts + metadata JSON saved per model, sklearn pinned to `1.7.2` for pickle compatibility.
- **Good training hygiene in Impl 2** (`train_bottleneck_classifier.py`): stratified 70/15/15 split, class weights, early stopping on `val_auc`, threshold tuned on *validation* (not test), `restore_best_weights`.
- **Shared preprocessing** (`prepare_features` in `srcML/nn_preprocessing/preprocessing.py`) reused across all models — avoids most training/serving skew.
- **Backend degrades gracefully** when a model artifact is missing; per-model + combined endpoints are a clean design.

---

## Critical issues

### 1. ⚠️ Row-level train/test split → disk-level leakage

All splits (Impl 0 notebook, Impl 1/2 AE splits, Impl 2 classifier split) are **per row, not per disk**.

**Why grouping by `serial_number` matters** (answer to "every file has only one unique disk"):
Backblaze publishes **one CSV per DAY**, and each daily file contains a snapshot of the **entire fleet** (~280k disks) — one row per disk per day. So the *same physical disk* (same `serial_number`) appears in up to 365 different files. That's why the dataset is 32M+ rows but only ~300k unique disks. When `build_dataset_from_many_csvs` / `build_balanced_dataset_from_csvs` samples healthy rows from many daily files, the same disk lands near-identical rows in both train and test after a row-level `train_test_split`. The model then partially "recognizes" disks it already saw → inflated ROC-AUC/recall.

- Verify yourself: count duplicate `serial_number` values in `sample.csv` — you'll find the same serials repeated across dates.
- Fix: `GroupShuffleSplit` / `StratifiedGroupKFold` grouped by `serial_number`; ideally also a time-based split (train Jan–Sep, test Oct–Dec).
- Failure rows are less affected (each failed disk contributes exactly 1 failure-day row), but healthy rows are heavily duplicated.

### 2. ⚠️ Zero-imputation of all-NaN failure rows → "missing = failure" artifact

The notebook output shows failure-day rows where **every SMART value is NaN** (disk died before the daily snapshot). `prepare_features`/`procesiraj_podatke` fill these with 0/medians → the classifier can learn "all zeros ⇒ failure", a signature that never occurs on a live disk scanned via `smartctl`.

- Audit what fraction of `csv/vseOdpovedi.csv` rows are all-NaN.
- Drop them or add explicit missingness-indicator features; re-report metrics.
- The HDBSCAN outlier cluster (66.9% failure rate) is likely dominated by these zero-imputed rows.

### 3. Balanced 50:50 **test set** metrics presented as real-world performance

**Clarification: training on a balanced set is fine and correct** — the issue is only about the *test/reporting* side:

- **Prevalence-invariant metrics** (safe to report from a balanced test set): recall, ROC-AUC, specificity.
- **Prevalence-dependent metrics** (NOT transferable from 50:50): precision, accuracy, F1, PR-AUC.

Concrete math: at ~1:1000 real-world daily failure prevalence, a model with 89% recall and ~89% precision on the 50:50 test set has real-world precision of roughly `0.89·0.001 / (0.89·0.001 + 0.11·0.999) ≈ 0.8%` — i.e. ~120 false alarms per true failure. Keep balanced training; just caveat the reported precision/accuracy/F1 as "balanced-set metrics", or additionally evaluate on a prevalence-realistic test set.

### 4. ⚠️ "Catches 9 of 10 failing disks *before* they die" — labels don't support this

Backblaze `failure=1` **only on the day of failure**. Current models detect *failure-day signatures*, not early warning.

**How to implement lookback labeling (if you want the "early warning" claim):**

1. Build a map `serial_number → failure_date` from `csv/vseOdpovedi.csv` (all 4,414 failures).
2. Scan the daily CSVs; for each row whose serial is in the map **and** whose date falls in `[failure_date − N, failure_date]` (N = 30 days typical), label it `1` ("will fail within N days"). This grows positives from 4,414 to up to ~130k rows — and these pre-failure rows have *real* SMART values, which also mitigates issue #2.
3. Healthy class = disks that **never fail during the whole year** (exclude the failing disks' rows older than the window, to avoid label noise).
4. Split **by serial_number** (mandatory here — the same failing disk now has up to 30 positive rows).
5. Evaluate as "predicted failure within N days" — this is the standard formulation in disk-failure-prediction literature and directly supports the README claim.

Practical route: extend `build_balanced_dataset_from_csvs` with a `lookback_days` parameter; Pass 1 collects failure serials + dates, Pass 2 selects windowed rows (daily file name gives the date).

### 5. Backend bug: HDBSCAN silently never loads in Docker — **needs fix**

`backend/main.py` lines ~64–66 load `clf_hdbscan.pkl` and `hdbscan_metadata.json` from `CLF_DIR` (`srcML/tensorflow_classification/`), but `umap_hdbscan.py` saves them to `srcML/tensorflow_clustering/`. Confirmed: the files exist only in `tensorflow_clustering/`. Result: `/api/predict/combined` silently runs with 3 models instead of 4.

**Fix** (one-liner): add `CLUSTER_DIR = APP_ROOT / "srcML" / "tensorflow_clustering"` and load both files from `CLUSTER_DIR` instead of `CLF_DIR`.

### 6. Cluster failure rates used as probabilities are inflated ~50×

**Detailed explanation:** `analyze_clusters` in `umap_hdbscan.py` computes each cluster's `failure_rate = failures / total` **on the balanced 50:50 dataset**, where the base failure rate is 50%. In reality the daily base rate is ~0.005–0.1%. A cluster's observed rate in balanced data is a *likelihood ratio* statement, not a real-world probability.

Bayes correction to a true prior π:

```
corrected = (r·π/0.5) / (r·π/0.5 + (1−r)·(1−π)/0.5)
```

Example: cluster rate r = 0.669 (outliers), true prior π = 0.001 → corrected ≈ 0.2%. So the "66.9% failure rate" cluster represents ~0.2% real-world risk — yet HIR feeds 0.669 into the RMS as if it were a probability, systematically pushing HIR up (weight 0.10).

Options:
- Recompute per-cluster rates on a prevalence-realistic sample; or
- Apply the Bayes prior-correction above; or
- Keep the rates as a **relative risk ranking only** (rescale to [0,1] by dividing by the max cluster rate) and document that the cluster component is ordinal, not probabilistic.

---

## Important (deferred by author — revisit later)

- **HIR ensemble never evaluated**: weights (0.3/0.4/0.2/0.1) hand-picked; the fused score is an RMS of quantities with different semantics (2 probabilities, a percentile-normalized error, a cluster rate). Run HIR on a held-out set + ablation vs Impl 2 alone.
- **Impl 0 notebook leakage**: median imputation and correlation-based feature dropping computed on the full dataset *before* the split. `disk_pipeline.py` hardcodes imputation medians (`0, 15, 1200, 12`) — persist actual training-time values in pickle/metadata instead.
- **Impl 1 threshold/eval overlap**: 99th-percentile threshold derived from val errors, then ROC/report use those same val rows as healthy negatives → optimistic. Use a third split.
- **Stage-1 AE / Stage-2 classifier data overlap**: both samplers use seed 42 over the same files → the encoder/scaler saw rows that end up in the classifier's test set.
- **MinMaxScaler + sigmoid on unbounded SMART counters** (`smart_9` hours, `smart_1` in millions): heavy tails squashed near 0; inference values outside the fitted range clip → reconstruction error measures range-clipping, not anomaly. Apply `log1p` before scaling in both AEs.

---

## Minor / hygiene

- **`disk_pipeline.py` clamp bug**: `if odstotek_tveganja > 95.0: odstotek_tveganja = 97.0` *raises* e.g. 95.1 → 97. Clamp bounds inconsistent (5–97 sklearn vs 3–97 HIR/backend); verdict thresholds duplicated in 3 files (`disk_pipeline.py`, `hir_final.py`, `backend/main.py`) — centralize.
- **Duplicated `FEATURE_COLUMNS`** in `tensorflow_anomaly/train_autoencoder.py` vs `preprocessing.py` — the local copy is dead code; drift risk.
- **`requirements.txt`**: `tensorflow` unpinned (pin it — `.keras` format breaks across versions); `multipart` is a wrong/unnecessary package (`python-multipart` already present).
- **CORS**: `allow_origins=["*"]` with `allow_credentials=True` is invalid/insecure — restrict to the frontend origin (relevant for a data-protection-themed demo).
- **Repo hygiene**: verify `sample.csv` (129 MB) and `srcML/izbolsani_podatki_vsi.csv` (32 MB) aren't git-tracked (`git ls-files -s sample.csv`) — `.gitignore` only helps if they were never added. `__pycache__/` present in tree.
- Misleading comments: "centriramo podatke glede na mediano" above a `MinMaxScaler`; README says the AE threshold is "learned" — it's a percentile of validation errors.

---

## Suggested priority order

1. Fix backend HDBSCAN path (5 min) — see #5.
2. Re-split by `serial_number`, re-train/re-evaluate Impl 0 & 2 — changes headline numbers (#1).
3. Audit/drop all-NaN failure rows, re-evaluate (#2).
4. Lookback labeling for the "before they die" claim (#4).
5. Correct/rescale cluster risk scores (#6).
6. Evaluate fused HIR + ablation.
7. Everything else.
