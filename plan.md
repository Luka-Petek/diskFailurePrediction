# Plan — AHI Evaluation + "Color Rock" Plot

## Goal

Compute AHI (Aggregated Health Index) for a sample of 100 disks (50 healthy, 50 failed) from the
cleaned dataset and visualize the result as a "color rock" scatter plot — two vertical columns
(Failure=0 and Failure=1) with each disk as a dot at its AHI% on the y-axis, colored green→red.
This shows how well AHI separates the two classes without needing any threshold logic.

---

## Terminology

- **HIR is fully renamed to AHI** everywhere in code, comments, and paper.
- Formula: `AHI = sqrt( Σ(wi · si²) / Σw ) × 100`  (RMS fusion of 4 model scores)
- Weights: sklearn RF=0.30, TF bottleneck clf=0.40, AE anomaly=0.20, HDBSCAN cluster=0.10
- Verdict bands: HEALTHY < 40, WARNING 40–75, CRITICAL > 75

---

## Files involved

| File | Role |
|---|---|
| `srcML/hir_final.py` | Core AHI computation — loads all 4 models and computes AHI for a single disk |
| `srcML/evaluate_ahi.py` | Batch evaluation — samples N disks from CSV, computes AHI per disk, saves CSV + plot |
| `srcML/ahi_final.py` | Thin CLI alias for `hir_final.py::predict_ahi` |
| `srcML/generate_hir_diagram.py` | Generates `Graphs/hir_formula.png` and `Graphs/ahi_formula.png` |
| `Graphs/ahi_color_rock.png` | Output figure (generated, ready) |
| `DiskJson/ahi_eval_sample.csv` | Per-disk AHI results (generated, ready) |
| `paper/main.tex` | Paper — figure needs to be integrated here |

---

## Data

- **Source**: `csv/koncniPodatkiZaModel.csv` — 8,828 rows, raw Backblaze format
  (`capacity_bytes`, `model`, `failure`, `smart_*_raw`).
- **In-sample**: same data used for model training. Fine for the visualization; note it in the paper.
- **Sample**: 50 healthy (failure=0) + 50 failed (failure=1) = 100 disks, balanced.
- **No holdout CSV available** — this is the only cleaned dataset with failure labels.

---

## Steps

### [DONE] Step 1 — Rename HIR → AHI
- `srcML/hir_final.py`: renamed `_compute_ahi`, `predict_ahi`
- `srcML/ahi_final.py`: created as clean CLI alias
- `srcML/generate_hir_diagram.py`: saves both `hir_formula.png` (legacy) and `ahi_formula.png`

### [DONE] Step 2 — Write `evaluate_ahi.py`
- Loads CSV, takes balanced sample (50+50)
- Loads all 4 model artifacts once
- Iterates per disk, calls all 4 scoring fns (each handles its own preprocessing — raw rows passed directly)
- Saves per-disk CSV to `DiskJson/ahi_eval_sample.csv`
- Generates color rock plot to `Graphs/ahi_color_rock.png`

Key design decisions:
- No `procesiraj_podatke` call before sampling — scoring functions do their own preprocessing internally.
- No threshold/confusion-matrix logic in the plot path.
- Per-disk progress printed to console.

### [DONE] Step 3 — Run and verify
Command used:
```
.venv\Scripts\python srcML/evaluate_ahi.py --data-csv csv/koncniPodatkiZaModel.csv --n-per-class 50
```
Results:
- `ahi_mean_healthy` = 34.18%  (mostly below WARNING line — correct)
- `ahi_mean_failed`  = 61.54%  (mostly in WARNING/CRITICAL band — correct)
- Clear visual separation between the two columns
- Plot saved to `Graphs/ahi_color_rock.png`

### [TODO] Step 4 — Integrate figure into `paper/main.tex`
- Add `\includegraphics` for `Graphs/ahi_color_rock.png`
- Write caption: describe the two columns, AHI separation, mean lines, zone bands, in-sample note
- Place it in the Results / Evaluation section

---

## How to re-run

```bash
.venv\Scripts\python srcML/evaluate_ahi.py \
  --data-csv csv/koncniPodatkiZaModel.csv \
  --n-per-class 50 \
  --random-state 42
```

Optional args:
- `--n-per-class N` — N disks per class (default 50, so 100 total)
- `--random-state N` — for reproducibility
- `--output-csv path` — where to save per-disk results
- `--plot-out path` — where to save the figure

---

## Notes

- The plot currently shows `(in-sample)` in the title — keep this for honesty in the paper.
- Mean lines per class are shown as colored horizontal bars for quick visual comparison.
- Zone lines at 40 and 75 are visual guides only, not classification thresholds.
- If a fresh holdout dataset becomes available later, re-run with `--data-csv <new_path>`.
