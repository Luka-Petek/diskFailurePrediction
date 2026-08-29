# srcML/tensorflow_clustering/analyze_cluster_features.py
#
# Analyzes per-cluster feature distributions using koncniPodatkiZaModel.csv
# and the trained HDBSCAN model. Prints a human-readable report and writes
# a template cluster_descriptions.json that you fill in manually.
#
# Usage:
#   python srcML/tensorflow_clustering/analyze_cluster_features.py

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLUSTERING_DIR = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from srcML.nn_preprocessing.preprocessing import prepare_features

CLF_DIR  = PROJECT_ROOT / "srcML" / "tensorflow_classification"
DATA_CSV = PROJECT_ROOT / "csv" / "koncniPodatkiZaModel.csv"
OUT_JSON = CLUSTERING_DIR / "cluster_descriptions.json"

KEY_FEATURES = [
    "smart_5_raw",    # reallocated sectors — key failure indicator
    "smart_197_raw",  # current pending sectors
    "smart_198_raw",  # uncorrectable errors
    "smart_187_raw",  # reported uncorrectable errors
    "smart_9_raw",    # power on hours (age)
    "smart_1_raw",    # read error rate
    "smart_188_raw",  # command timeout
    "smart_192_raw",  # power off retract count
    "smart_193_raw",  # load/unload cycles
    "capacity_gigabytes",
    "jeSSD",
]

#glede na vsako gruco bomo naredili feature importance + napisali opis npr: "high reallicated sectors, >10TB,..."
def load_artifacts():
    encoder = tf.keras.models.load_model(CLF_DIR / "disk_clf_encoder.keras")
    scaler  = joblib.load(CLF_DIR / "clf_scaler.pkl")
    clusterer = joblib.load(CLUSTERING_DIR / "clf_hdbscan.pkl")
    with open(CLUSTERING_DIR / "hdbscan_metadata.json", encoding="utf-8") as f:
        meta = json.load(f)
    return encoder, scaler, clusterer, meta


def assign_clusters(df_raw: pd.DataFrame, encoder, scaler, clusterer) -> np.ndarray:
    import hdbscan as hdbscan_lib
    X = prepare_features(df_raw)
    X_scaled   = scaler.transform(X).astype("float32")
    bottleneck = encoder.predict(X_scaled, batch_size=512, verbose=0)
    labels, _  = hdbscan_lib.approximate_predict(clusterer, bottleneck)
    return labels


def top_distinguishing_features(df: pd.DataFrame, cluster_mask: np.ndarray, feat_cols: list, top_n: int = 4) -> str:
    """Returns a short string of features whose mean in this cluster differs most from global mean."""
    global_means = df[feat_cols].mean()
    cluster_means = df.loc[cluster_mask, feat_cols].mean()
    # normalised difference: (cluster_mean - global_mean) / (global_std + 1e-9)
    global_stds = df[feat_cols].std() + 1e-9
    diff = (cluster_means - global_means) / global_stds
    top = diff.abs().nlargest(top_n)
    parts = []
    for feat in top.index:
        d = diff[feat]
        arrow = "↑" if d > 0 else "↓"
        parts.append(f"{feat} {arrow}")
    return ",  ".join(parts)


def main():
    try:
        import hdbscan  # noqa: F401
    except ImportError:
        print("Napaka: pip install hdbscan")
        sys.exit(1)

    print(f"Berem {DATA_CSV} ...")
    df = pd.read_csv(DATA_CSV, low_memory=False)
    print(f"  Vrstic: {len(df):,}  |  failure=1: {df['failure'].sum():,}")

    encoder, scaler, clusterer, meta = load_artifacts()

    print("Dodeljevanje clusterjev (approximate_predict)...")
    cluster_ids = assign_clusters(df, encoder, scaler, clusterer)
    df["_cluster"] = cluster_ids

    cluster_risk = meta["cluster_risk"]
    feat_cols = [c for c in KEY_FEATURES if c in df.columns]

    print("\n" + "="*72)
    print(f"{'CLUSTER':>8}  {'RISK_LABEL':15}  {'fail%':>6}  {'n':>6}  TOP DISTINGUISHING FEATURES")
    print("="*72)

    template = {}

    for cid_str in sorted(cluster_risk.keys(), key=lambda x: int(x)):
        cid = int(cid_str)
        info = cluster_risk[cid_str]
        mask = df["_cluster"] == cid
        n_in_csv = int(mask.sum())

        if n_in_csv == 0:
            top_feats = "(no rows in this CSV)"
        else:
            top_feats = top_distinguishing_features(df, mask, feat_cols)

        # extra quick stats
        pct_ssd = float(df.loc[mask, "jeSSD"].mean() * 100) if "jeSSD" in df.columns and n_in_csv > 0 else float("nan")
        mean_cap = float(df.loc[mask, "capacity_gigabytes"].mean()) if "capacity_gigabytes" in df.columns and n_in_csv > 0 else float("nan")
        mean_s5  = float(df.loc[mask, "smart_5_raw"].mean())  if "smart_5_raw" in df.columns and n_in_csv > 0 else float("nan")
        mean_s197= float(df.loc[mask, "smart_197_raw"].mean()) if "smart_197_raw" in df.columns and n_in_csv > 0 else float("nan")

        label = "OUTLIER" if cid == -1 else info["risk_label"]
        print(f"{cid_str:>8}  {label:15}  {info['failure_rate']*100:>5.1f}%  {n_in_csv:>6}  {top_feats}")
        print(f"          {'':15}  SSD={pct_ssd:.0f}%  cap={mean_cap:.0f}GB  smart5={mean_s5:.1f}  smart197={mean_s197:.1f}")

        template[cid_str] = {
            "risk_label":   label,
            "failure_rate": info["failure_rate"],
            "n_samples":    info["total_samples"],
            "pct_ssd":      round(pct_ssd, 1),
            "mean_capacity_gb": round(mean_cap, 0),
            "mean_smart5_reallocated": round(mean_s5, 1),
            "mean_smart197_pending":   round(mean_s197, 1),
            "top_features": top_feats,
            "description":  "FILL IN",
        }

    print("="*72)

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(template, f, indent=2, ensure_ascii=False)
    print(f"\nTemplate shranjen: {OUT_JSON}")
    print("Odpri ga in zamenjaj vsak 'FILL IN' s kratkim opisom.")


if __name__ == "__main__":
    main()
