from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure project root is on sys.path so `srcML` can be imported when run as a script
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from srcML.hir_final import (
    PROJECT_ROOT,
    _load_sklearn_pipeline,
    _load_tf_clf_artifacts,
    _load_anomaly_artifacts,
    _load_clustering_artifacts,
    _score_sklearn,
    _score_tf_clf,
    _score_anomaly,
    _score_clustering,
    _compute_ahi,
)
from srcML.nn_preprocessing.preprocessing import build_balanced_dataset_from_csvs


def _read_csv(csv_path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(csv_path, low_memory=False)
    except Exception:
        return pd.read_csv(csv_path, sep=";", low_memory=False)


def _balanced_sample(df: pd.DataFrame, n_per_class: int, random_state: int) -> pd.DataFrame:
    failed  = df[df["failure"] == 1]
    healthy = df[df["failure"] == 0]
    n_f = min(n_per_class, len(failed))
    n_h = min(n_per_class, len(healthy))
    part_f = failed.sample(n=n_f, random_state=random_state)
    part_h = healthy.sample(n=n_h, random_state=random_state + 1)
    return pd.concat([part_h, part_f], ignore_index=True)


def _compute_ahi_for_row(raw_row: pd.DataFrame,
                          sklearn_pipeline,
                          encoder, classifier, clf_scaler,
                          ae_model, ae_scaler, ae_meta,
                          clusterer, cluster_meta) -> dict:
    # Each scoring function handles its own preprocessing internally — pass raw row directly
    s_skl         = _score_sklearn(sklearn_pipeline, raw_row)
    s_clf         = _score_tf_clf(encoder, classifier, clf_scaler, raw_row)
    s_an          = _score_anomaly(ae_model, ae_scaler, ae_meta, raw_row)
    cluster_result = _score_clustering(clusterer, cluster_meta, encoder, clf_scaler, raw_row)
    result = _compute_ahi(s_skl, s_clf, s_an, cluster_result["score"])
    result["cluster_info"] = {
        "cluster_id":  cluster_result["cluster_id"],
        "risk_label":  cluster_result["risk_label"],
        "risk_score":  round(cluster_result["score"], 4),
        "description": cluster_result["description"],
    }
    return result


def evaluate_ahi(
    sample: pd.DataFrame,
    sklearn_dir: Path,
    clf_dir: Path,
    anomaly_dir: Path,
    clustering_dir: Path,
    random_state: int,
    output_csv: Optional[Path],
    plot_out: Optional[Path],
    dataset_label: str,
) -> dict:
    print(f"Vzorec: {len(sample)} diskov  ({sample['failure'].sum():.0f} failed, {(sample['failure']==0).sum():.0f} healthy)")

    print("Nalagam artefakte...")
    sklearn_pipeline = _load_sklearn_pipeline(sklearn_dir)
    encoder, classifier, clf_scaler, clf_meta = _load_tf_clf_artifacts(clf_dir)
    ae_model, ae_scaler, ae_meta = _load_anomaly_artifacts(anomaly_dir)
    try:
        clusterer, cluster_meta = _load_clustering_artifacts(clustering_dir)
    except Exception as e:
        print(f"[OPOZORILO] Clustering ni na voljo ({e}), fallback = 0.5")
        clusterer, cluster_meta = None, {"cluster_risk": {"-1": {"risk_score": 0.5}}}

    rows = []
    total = len(sample)
    for i in range(total):
        raw_row = sample.iloc[[i]]   # single-row DataFrame, raw columns — preprocessing happens inside scoring fns
        label = int(raw_row["failure"].iloc[0])
        print(f"  Disk {i+1:>3}/{total}  (failure={label})", end="\r", flush=True)

        fused = _compute_ahi_for_row(
            raw_row,
            sklearn_pipeline,
            encoder, classifier, clf_scaler,
            ae_model, ae_scaler, ae_meta,
            clusterer, cluster_meta,
        )
        rows.append({
            "disk_idx":  i,
            "label":     label,
            "ahi":       fused["ahi_score"],
            "verdict":   fused["verdict"],
            "sklearn_failure_prob": fused["components"]["sklearn_failure_prob"],
            "tf_clf_failure_prob":  fused["components"]["tf_clf_failure_prob"],
            "anomaly_score":        fused["components"]["anomaly_score"],
            "cluster_risk_score":   fused["components"]["cluster_risk_score"],
        })

    print(f"\nDone — {total} diskov ocenjenih.")
    result_df = pd.DataFrame(rows)

    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(output_csv, index=False)
        print(f"Rezultati shranjeni: {output_csv}")

    if plot_out is not None:
        _plot_color_rock(result_df, plot_out, dataset_label)
        print(f"Graf shranjen: {plot_out}")

    summary = {
        "sample_size": int(len(result_df)),
        "n_failed":  int(result_df["label"].sum()),
        "n_healthy": int((result_df["label"] == 0).sum()),
        "ahi_mean_failed":  round(float(result_df[result_df["label"]==1]["ahi"].mean()), 2),
        "ahi_mean_healthy": round(float(result_df[result_df["label"]==0]["ahi"].mean()), 2),
        "output_csv": str(output_csv) if output_csv else None,
        "plot":       str(plot_out)   if plot_out   else None,
    }
    return summary


def _plot_color_rock(df: pd.DataFrame, out_path: Path, dataset_label: str = "in-sample") -> None:
    BG   = "#111111"
    rng  = np.random.default_rng(seed=0)

    labels = df["label"].astype(int).to_numpy()
    ahi    = df["ahi"].to_numpy()

    # Jitter x so dots don't overlap vertically
    jitter = rng.uniform(-0.18, 0.18, size=len(labels))
    x_pos  = labels.astype(float) + jitter

    fig, ax = plt.subplots(figsize=(7, 6), facecolor=BG)
    ax.set_facecolor(BG)

    sc = ax.scatter(
        x_pos, ahi,
        c=ahi, cmap="RdYlGn_r",
        vmin=0, vmax=100,
        s=22, alpha=0.88, linewidths=0,
    )

    # Zone lines (visual guides, not thresholds)
    for level, color, label in [
        (40, "#f0c419", "WARNING  40"),
        (75, "#ff5555", "CRITICAL  75"),
    ]:
        ax.axhline(level, color=color, lw=1.0, ls="--", alpha=0.65)
        ax.text(1.44, level, label, color=color, va="center", ha="right", fontsize=7.5)

    # Axes
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(0, 100)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Failure = 0\n(Healthy)", "Failure = 1\n(Failed)"],
                       color="#dddddd", fontsize=11)
    ax.set_ylabel("AHI  (%)", color="#dddddd", fontsize=11)
    ax.tick_params(colors="#aaaaaa")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")

    # Color bar
    cbar = plt.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("AHI  (%)", color="#dddddd", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="#aaaaaa")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#cccccc")

    # Means per class
    for lbl, color in [(0, "#55dd88"), (1, "#ff7777")]:
        mean_val = float(df[df["label"] == lbl]["ahi"].mean())
        ax.hlines(mean_val, lbl - 0.35, lbl + 0.35,
                  colors=color, lw=2.0, alpha=0.9, zorder=5)
        ax.text(lbl + 0.37, mean_val, f"mean {mean_val:.1f}",
                color=color, va="center", fontsize=8)

    n_h = int((df["label"] == 0).sum())
    n_f = int((df["label"] == 1).sum())
    ax.set_title(
        f"AHI vs. actual disk failure  (n={n_h+n_f}, {dataset_label})",
        color="#eeeeee", fontsize=11, pad=10,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate AHI and create a color rock plot.")
    p.add_argument("--data-csv",   type=str, default=None,
                   help="Path to a single CSV file with failure column.")
    p.add_argument("--data-dir",   type=str, default=None,
                   help="Path to a directory of CSV files (e.g. DiskData2023). Scans recursively.")
    p.add_argument("--n-per-class", type=int, default=50,
                   help="Number of disks per class (failure=0 and failure=1). Default 50 → 100 total.")
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--output-csv",  type=str,
                   default=str(PROJECT_ROOT / "DiskJson" / "ahi_eval_sample.csv"))
    p.add_argument("--plot-out",    type=str,
                   default=str(PROJECT_ROOT / "Graphs" / "ahi_color_rock.png"))
    p.add_argument("--sklearn-dir", type=str,
                   default=str(PROJECT_ROOT / "srcML" / "sklearn"))
    p.add_argument("--clf-dir",     type=str,
                   default=str(PROJECT_ROOT / "srcML" / "tensorflow_classification"))
    p.add_argument("--anomaly-dir", type=str,
                   default=str(PROJECT_ROOT / "srcML" / "tensorflow_anomaly"))
    p.add_argument("--clustering-dir", type=str,
                   default=str(PROJECT_ROOT / "srcML" / "tensorflow_clustering"))

    args = p.parse_args()

    if not args.data_csv and not args.data_dir:
        p.error("Provide either --data-csv or --data-dir")

    if args.data_dir:
        data_dir = Path(args.data_dir)
        print(f"Branje CSV datotek iz: {data_dir}")
        healthy_df, failure_df = build_balanced_dataset_from_csvs(
            data_dir=data_dir,
            max_failure=args.n_per_class,
            random_state=args.random_state,
        )
        sample = pd.concat([healthy_df, failure_df], ignore_index=True)
        dataset_label = f"holdout 2023"
    else:
        csv_path = Path(args.data_csv)
        df = _read_csv(csv_path)
        if df.empty:
            raise RuntimeError(f"CSV je prazen: {csv_path}")
        if "failure" not in df.columns:
            raise RuntimeError("CSV nima stolpca 'failure'.")
        sample = _balanced_sample(df, n_per_class=args.n_per_class, random_state=args.random_state)
        dataset_label = "in-sample"

    summary = evaluate_ahi(
        sample=sample,
        sklearn_dir=Path(args.sklearn_dir),
        clf_dir=Path(args.clf_dir),
        anomaly_dir=Path(args.anomaly_dir),
        clustering_dir=Path(args.clustering_dir),
        random_state=args.random_state,
        output_csv=Path(args.output_csv),
        plot_out=Path(args.plot_out),
        dataset_label=dataset_label,
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
