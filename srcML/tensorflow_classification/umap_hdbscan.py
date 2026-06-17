#python srcML/tensorflow_classification/umap_hdbscan.py --data-dir DiskData
#Zahteva: pip install umap-learn hdbscan

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent
GRAPHS_DIR = PROJECT_ROOT / "Graphs"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from srcML.nn_preprocessing.preprocessing import (
    build_balanced_dataset_from_csvs,
    prepare_features,
)

DEFAULT_FAILURE_CSV = PROJECT_ROOT / "csv" / "vseOdpovedi.csv"


def load_encoder_artifacts(clf_dir: Path) -> tuple:
    encoder = tf.keras.models.load_model(clf_dir / "disk_clf_encoder.keras")
    scaler = joblib.load(clf_dir / "clf_scaler.pkl")
    return encoder, scaler


def extract_bottleneck(encoder, scaler, df_raw: pd.DataFrame) -> np.ndarray:
    X = prepare_features(df_raw)
    X_scaled = scaler.transform(X).astype("float32")
    bottleneck = encoder.predict(X_scaled, batch_size=4096, verbose=0)
    return bottleneck


def analyze_clusters(hdbscan_labels: np.ndarray, labels_true: np.ndarray) -> dict:
    metadata = {}
    unique_clusters = np.unique(hdbscan_labels)

    for cid in unique_clusters:
        mask = hdbscan_labels == cid
        total = int(mask.sum())
        failures = int(labels_true[mask].sum())
        failure_rate = failures / total if total > 0 else 0.0

        if cid == -1:
            risk_label = "OUTLIER"
        elif failure_rate > 0.50:
            risk_label = "HIGH_RISK"
        elif failure_rate >= 0.10:
            risk_label = "ELEVATED_RISK"
        else:
            risk_label = "LOW_RISK"
        risk_score = round(failure_rate, 4)

        metadata[str(cid)] = {
            "risk_label": risk_label,
            "risk_score": risk_score,
            "failure_rate": round(failure_rate, 4),
            "total_samples": total,
            "failure_samples": failures,
            "is_outlier": bool(cid == -1),
        }

    return metadata


def plot_umap_hdbscan(
    umap_coords: np.ndarray,
    hdbscan_labels: np.ndarray,
    labels_true: np.ndarray,
    graphs_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    unique_clusters = np.unique(hdbscan_labels)
    n_clusters = len(unique_clusters)
    cmap = plt.cm.get_cmap("tab20", max(n_clusters, 1))
    cluster_color_map = {cid: ("gray" if cid == -1 else cmap(i)) for i, cid in enumerate(unique_clusters)}

    colors_cluster = [cluster_color_map[c] for c in hdbscan_labels]
    axes[0].scatter(umap_coords[:, 0], umap_coords[:, 1], c=colors_cluster, alpha=0.4, s=5)
    axes[0].set_title(f"HDBSCAN Clusters (UMAP 2D) — {n_clusters} clusters")
    axes[0].set_xlabel("UMAP-1")
    axes[0].set_ylabel("UMAP-2")

    from matplotlib.patches import Patch
    legend_handles = [Patch(color=cluster_color_map[c], label=f"Cluster {c}" if c != -1 else "Outlier") for c in unique_clusters]
    axes[0].legend(handles=legend_handles, markerscale=2, loc="best", fontsize=7)

    colors_label = ["steelblue" if l == 0 else "crimson" for l in labels_true]
    axes[1].scatter(umap_coords[:, 0], umap_coords[:, 1], c=colors_label, alpha=0.3, s=5)
    axes[1].set_title("Failure vs Healthy (UMAP 2D)")
    axes[1].set_xlabel("UMAP-1")
    axes[1].set_ylabel("UMAP-2")
    axes[1].legend(handles=[Patch(color="steelblue", label="Healthy"), Patch(color="crimson", label="Failure")])

    plt.tight_layout()
    graphs_dir.mkdir(parents=True, exist_ok=True)
    save_path = graphs_dir / "umap_hdbscan.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"UMAP/HDBSCAN plot shranjen: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="UMAP + HDBSCAN clustering na Impl 2 bottleneck features.")
    parser.add_argument("--data-dir", type=str, default=str(PROJECT_ROOT / "DiskData"), help="Mapa z DiskData CSV-ji.")
    parser.add_argument("--max-failure", type=int, default=None, help="Maks. stevilo failure vrstic (default=vse razpolozljive).")
    parser.add_argument("--failure-csv", type=str,
                        default=str(DEFAULT_FAILURE_CSV) if DEFAULT_FAILURE_CSV.exists() else None,
                        help="Pot do CSV z vsemi failure vrsticami (preskoce Pass 1).")
    parser.add_argument("--umap-neighbors", type=int, default=30, help="UMAP n_neighbors.")
    parser.add_argument("--umap-min-dist", type=float, default=0.1, help="UMAP min_dist.")
    parser.add_argument("--hdbscan-min-cluster-size", type=int, default=50, help="HDBSCAN min_cluster_size.")
    parser.add_argument("--clf-dir", type=str, default=str(OUTPUT_DIR), help="Mapa z Impl 2 artefakti.")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    try:
        import umap
        import hdbscan as hdbscan_lib
    except ImportError:
        print("Napaka: namestitvi manjkajoci paketi z:  pip install umap-learn hdbscan")
        sys.exit(1)

    data_dir = Path(args.data_dir)
    clf_dir = Path(args.clf_dir)
    graphs_dir = GRAPHS_DIR
    failure_csv = Path(args.failure_csv) if args.failure_csv else None

    print("Nalagam encoder in scaler...")
    encoder, scaler = load_encoder_artifacts(clf_dir)

    print(f"Berem podatke iz: {data_dir}")
    healthy_df, failure_df = build_balanced_dataset_from_csvs(
        data_dir=data_dir,
        max_failure=args.max_failure,
        random_state=args.random_state,
        failure_csv=failure_csv,
    )

    print(f"Zdravih vrstic: {len(healthy_df):,} | Failure vrstic: {len(failure_df):,}")

    all_df = pd.concat([healthy_df, failure_df], ignore_index=True)
    labels_true = np.array(
        [0] * len(healthy_df) + [1] * len(failure_df), dtype=np.int32
    )

    print("Ekstrakcija bottleneck features...")
    bottleneck = extract_bottleneck(encoder, scaler, all_df)
    print(f"Bottleneck shape: {bottleneck.shape}")

    print(f"UMAP redukcija {bottleneck.shape[1]}D → 2D (n_neighbors={args.umap_neighbors}, min_dist={args.umap_min_dist})...")
    reducer = umap.UMAP(
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        metric="euclidean",
        random_state=42,
        verbose=True,
    )
    umap_coords = reducer.fit_transform(bottleneck)
    print(f"UMAP done. Shape: {umap_coords.shape}")

    print(f"HDBSCAN clustering (min_cluster_size={args.hdbscan_min_cluster_size})...")
    clusterer = hdbscan_lib.HDBSCAN(
        min_cluster_size=args.hdbscan_min_cluster_size,
        metric="euclidean",
        prediction_data=True,
    )
    hdbscan_labels = clusterer.fit_predict(bottleneck)

    n_clusters = len(set(hdbscan_labels)) - (1 if -1 in hdbscan_labels else 0)
    n_outliers = int((hdbscan_labels == -1).sum())
    print(f"HDBSCAN: {n_clusters} clusterjev, {n_outliers:,} outlierjev ({n_outliers/len(hdbscan_labels):.1%})")

    cluster_metadata = analyze_clusters(hdbscan_labels, labels_true)

    print("\nCluster analiza:")
    for cid, info in sorted(cluster_metadata.items(), key=lambda x: int(x[0])):
        marker = " ← OUTLIER" if info["is_outlier"] else ""
        print(f"  Cluster {cid}: {info['risk_label']} | failure_rate={info['failure_rate']:.2%} | n={info['total_samples']:,}{marker}")

    #Shranjevanje metadata
    full_metadata = {
        "n_clusters": n_clusters,
        "n_outliers": n_outliers,
        "outlier_ratio": round(n_outliers / len(hdbscan_labels), 4),
        "bottleneck_dim": int(bottleneck.shape[1]),
        "umap_params": {
            "n_neighbors": args.umap_neighbors,
            "min_dist": args.umap_min_dist,
            "metric": "euclidean",
        },
        "hdbscan_params": {
            "min_cluster_size": args.hdbscan_min_cluster_size,
            "metric": "euclidean",
        },
        "cluster_risk": cluster_metadata,
    }

    meta_path = clf_dir / "hdbscan_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(full_metadata, f, indent=2, ensure_ascii=False)
    print(f"\nMetadata shranjena: {meta_path}")

    #Shranimo HDBSCAN model za inference (approximate_predict na novih točkah)
    hdbscan_path = clf_dir / "clf_hdbscan.pkl"
    joblib.dump(clusterer, hdbscan_path)
    print(f"HDBSCAN model shranjen: {hdbscan_path}")

    #Shranimo UMAP reducer za re-vizualizacijo (ni potreben za inference)
    umap_path = clf_dir / "clf_umap_reducer.pkl"
    joblib.dump(reducer, umap_path)
    print(f"UMAP reducer shranjen: {umap_path}")

    plot_umap_hdbscan(umap_coords, hdbscan_labels, labels_true, graphs_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
