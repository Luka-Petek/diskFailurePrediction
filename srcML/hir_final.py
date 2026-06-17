#  python srcML/hir_final.py --input DiskJson/disk_data_sda.json
#
#  HIR (Health Index Rating) — kombinirana formula vseh 4 modelov:
#    Impl 0  — Sklearn Random Forest        (utez 0.30)
#    Impl 2  — TF Bottleneck Classifier     (utez 0.40)  ← najboljsi rezultati
#    Impl 1  — TF Anomaly Detection AE      (utez 0.20)
#    Impl C  — TF Clustering HDBSCAN        (utez 0.10)

import sys
import os

# POMEMBNO: Python avtomatsko doda srcML/ v sys.path[0] ko tecemo ta skript.
# To povzroci da lokalni srcML/sklearn/ paket zasenči scikit-learn sklearn.
# Odstranimo script dir PRED katerimkoli importom.
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir in sys.path:
    sys.path.remove(_script_dir)

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import sklearn
import sklearn.ensemble
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from srcML.sklearn.disk_pipeline import pretvori_json_v_surovi_df
from srcML.nn_preprocessing.preprocessing import prepare_features

W_SKLEARN  = 0.30   # RF na 19-dim + manufacturer
W_TF_CLF   = 0.40   # AE encoder → 8-dim → classifier  (ROC-AUC 0.929, F1 0.887)
W_ANOMALY  = 0.20   # rekonstrukcijska napaka AE        (ROC-AUC 0.901, recall 0.447)
W_CLUSTER  = 0.10   # HDBSCAN cluster failure_rate

def _load_sklearn_pipeline(sklearn_dir: Path):
    return joblib.load(sklearn_dir / "disk_health_pipeline.pkl")

def _load_tf_clf_artifacts(clf_dir: Path) -> tuple:
    encoder    = tf.keras.models.load_model(clf_dir / "disk_clf_encoder.keras")
    classifier = tf.keras.models.load_model(clf_dir / "disk_bottleneck_classifier.keras")
    scaler     = joblib.load(clf_dir / "clf_scaler.pkl")
    with open(clf_dir / "bottleneck_metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)
    return encoder, classifier, scaler, metadata

def _load_anomaly_artifacts(anomaly_dir: Path) -> tuple:
    model  = tf.keras.models.load_model(anomaly_dir / "disk_autoencoder.keras")
    scaler = joblib.load(anomaly_dir / "tf_scaler.pkl")
    with open(anomaly_dir / "tf_metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)
    return model, scaler, metadata

def _load_clustering_artifacts(clustering_dir: Path) -> tuple:
    clusterer = joblib.load(clustering_dir / "clf_hdbscan.pkl")
    with open(clustering_dir / "hdbscan_metadata.json", encoding="utf-8") as f:
        cluster_meta = json.load(f)
    return clusterer, cluster_meta

def _score_sklearn(pipeline, raw_df: "pd.DataFrame") -> float:
    result = pipeline.analyze(raw_df)
    return float(result["failure_probability"])

def _score_tf_clf(encoder, classifier, scaler, raw_df: "pd.DataFrame") -> float:
    X = prepare_features(raw_df)
    X_scaled   = scaler.transform(X).astype("float32")
    bottleneck = encoder.predict(X_scaled, batch_size=1, verbose=0)
    prob       = float(classifier.predict(bottleneck, batch_size=1, verbose=0).flatten()[0])
    return prob

def _score_anomaly(ae_model, scaler, metadata, raw_df: "pd.DataFrame") -> float:
    X = prepare_features(raw_df)
    X_scaled     = scaler.transform(X).astype("float32")
    reconstructed = ae_model.predict(X_scaled, batch_size=1, verbose=0)
    error        = float(np.mean(np.abs(X_scaled - reconstructed)))

    threshold = metadata["threshold"]
    p999      = metadata["normalization"]["score_p999"]

    if p999 <= threshold:
        return 0.0
    return float(np.clip((error - threshold) / (p999 - threshold), 0.0, 1.0))

def _score_clustering(
    clusterer, cluster_meta,
    encoder, scaler,
    raw_df: "pd.DataFrame",
) -> float:
    try:
        import hdbscan as hdbscan_lib
    except ImportError:
        return float(cluster_meta["cluster_risk"].get("-1", {}).get("risk_score", 0.5))

    try:
        X = prepare_features(raw_df)
        X_scaled   = scaler.transform(X).astype("float32")
        bottleneck = encoder.predict(X_scaled, batch_size=1, verbose=0)

        labels, _strengths     = hdbscan_lib.approximate_predict(clusterer, bottleneck)
        cluster_id             = str(int(labels[0]))

        cluster_risks = cluster_meta["cluster_risk"]
        if cluster_id in cluster_risks:
            return float(cluster_risks[cluster_id]["risk_score"])
        #  neznan cluster → fallback na outlier risk
        return float(cluster_risks.get("-1", {}).get("risk_score", 0.5))

    except Exception as exc:
        print(f"[OPOZORILO] Clustering scoring odpovedan ({exc}), fallback = 0.5", file=sys.stderr)
        return 0.5

def _compute_hir(
    sklearn_prob: float,
    tf_clf_prob: float,
    anomaly_score: float,
    cluster_score: float,
) -> dict:
    #  RMS formula: sqrt( Σ(wi · si²) / Σw )  — ker Σw = 1.0, Σw odpade
    rms_score = np.sqrt(
        W_SKLEARN * sklearn_prob  ** 2
        + W_TF_CLF  * tf_clf_prob   ** 2
        + W_ANOMALY * anomaly_score ** 2
        + W_CLUSTER * cluster_score ** 2
    )

    hir = float(np.clip(rms_score * 100, 3.0, 97.0))

    if hir >= 75.0:
        verdict = "CRITICAL"
    elif hir >= 40.0:
        verdict = "WARNING"
    else:
        verdict = "HEALTHY"

    return {
        "hir_score":  round(hir, 2),
        "verdict":    verdict,
        "components": {
            "sklearn_failure_prob":  round(sklearn_prob,   4),
            "tf_clf_failure_prob":   round(tf_clf_prob,    4),
            "anomaly_score":         round(anomaly_score,  4),
            "cluster_risk_score":    round(cluster_score,  4),
        },
        "weights": {
            "sklearn":  W_SKLEARN,
            "tf_clf":   W_TF_CLF,
            "anomaly":  W_ANOMALY,
            "cluster":  W_CLUSTER,
        },
    }

def predict_hir(
    smartctl_json_path: Path,
    sklearn_dir: Path,
    clf_dir: Path,
    anomaly_dir: Path,
    clustering_dir: Path,
) -> dict:
    print("Nalagam artefakte...", file=sys.stderr)
    pipeline                         = _load_sklearn_pipeline(sklearn_dir)
    encoder, classifier, clf_scaler, clf_meta = _load_tf_clf_artifacts(clf_dir)
    ae_model, ae_scaler, ae_meta     = _load_anomaly_artifacts(anomaly_dir)
    clusterer, cluster_meta = _load_clustering_artifacts(clustering_dir)

    with open(smartctl_json_path, encoding="utf-8") as f:
        smartctl_dict = json.load(f)
    raw_df = pretvori_json_v_surovi_df(smartctl_dict)

    print("Racunam signale...", file=sys.stderr)
    sklearn_prob  = _score_sklearn(pipeline, raw_df)
    tf_clf_prob   = _score_tf_clf(encoder, classifier, clf_scaler, raw_df)
    anomaly_score = _score_anomaly(ae_model, ae_scaler, ae_meta, raw_df)
    cluster_score = _score_clustering(
        clusterer, cluster_meta,
        encoder, clf_scaler,          # encoder se deli z TF clf
        raw_df,
    )

    result = _compute_hir(sklearn_prob, tf_clf_prob, anomaly_score, cluster_score)

    result["model_metadata"] = {
        "tf_clf":  {
            "roc_auc":       clf_meta["evaluation"]["roc_auc"],
            "failure_f1":    clf_meta["evaluation"]["failure_f1"],
            "bottleneck_dim": clf_meta["bottleneck_dim"],
        },
        "anomaly": {
            "roc_auc":           ae_meta["evaluation"]["roc_auc"],
            "failure_recall":    ae_meta["evaluation"]["classification_report"]["failure"]["recall"],
            "threshold_pct":     ae_meta["threshold_percentile"],
        },
        "clustering": {
            "n_clusters":    cluster_meta["n_clusters"],
            "outlier_ratio": cluster_meta["outlier_ratio"],
        },
    }

    return result

def main() -> None:
    parser = argparse.ArgumentParser(
        description="HIR — kombinirana napoved zdravja diska (4 modeli)."
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Pot do smartctl JSON datoteke (smartctl -A -i -j).",
    )
    parser.add_argument(
        "--sklearn-dir", type=str,
        default=str(PROJECT_ROOT / "srcML" / "sklearn"),
        help="Mapa z disk_health_pipeline.pkl.",
    )
    parser.add_argument(
        "--clf-dir", type=str,
        default=str(PROJECT_ROOT / "srcML" / "tensorflow_classification"),
        help="Mapa z Impl 2 artefakti (encoder, classifier, scaler, metadata).",
    )
    parser.add_argument(
        "--anomaly-dir", type=str,
        default=str(PROJECT_ROOT / "srcML" / "tensorflow_anomaly"),
        help="Mapa z Impl 1 artefakti (autoencoder, scaler, metadata).",
    )
    parser.add_argument(
        "--clustering-dir", type=str,
        default=str(PROJECT_ROOT / "srcML" / "tensorflow_clustering"),
        help="Mapa s clustering artefakti (hdbscan, umap_reducer, metadata).",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Napaka: datoteka ne obstaja: {input_path}", file=sys.stderr)
        sys.exit(1)

    result = predict_hir(
        smartctl_json_path=input_path,
        sklearn_dir=Path(args.sklearn_dir),
        clf_dir=Path(args.clf_dir),
        anomaly_dir=Path(args.anomaly_dir),
        clustering_dir=Path(args.clustering_dir),
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
