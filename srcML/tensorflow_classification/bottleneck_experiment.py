#python srcML/tensorflow_classification/bottleneck_experiment.py --data-dir DiskData

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DISKJSON_DIR = PROJECT_ROOT / "DiskJson"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from srcML.tensorflow_classification.train_autoencoder import build_autoencoder
from srcML.nn_preprocessing.preprocessing import (
    build_dataset_from_many_csvs,
    prepare_features,
)
from srcML.tensorflow_classification.train_bottleneck_classifier import (
    build_classifier,
    extract_bottleneck_features,
    find_best_threshold,
)

DEFAULT_DIMS = [4, 6, 7, 8, 10, 12]


def train_and_evaluate_dim(
    dim: int,
    X_train_scaled: np.ndarray,
    X_val_scaled: np.ndarray,
    X_failed_scaled: np.ndarray,
    ae_epochs: int,
    random_state: int,
) -> dict:

    #treniranje autoencoderja za to dimenzijo
    ae = build_autoencoder(input_dim=X_train_scaled.shape[1], bottleneck_dim=dim)
    ae.fit(
        X_train_scaled, X_train_scaled,
        validation_data=(X_val_scaled, X_val_scaled),
        epochs=ae_epochs,
        batch_size=128,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True
            )
        ],
        verbose=0,
    )

    #ekstrakcija encoder submodela
    encoder = tf.keras.Model(
        inputs=ae.input,
        outputs=ae.get_layer("bottleneck").output,
    )
    encoder.trainable = False

    #ekstrakcija bottleneck features
    Z_healthy_train = extract_bottleneck_features(encoder, X_train_scaled)
    Z_healthy_val = extract_bottleneck_features(encoder, X_val_scaled)
    Z_failed_all = extract_bottleneck_features(encoder, X_failed_scaled)

    #razdelitev okvarjenih na train/val del (da ni data leakage)
    Z_failed_train, Z_failed_val = train_test_split(
        Z_failed_all, test_size=0.30, random_state=random_state
    )

    #sestavimo označeni dataset
    Z_train_clf = np.concatenate([Z_healthy_train, Z_failed_train])
    y_train_clf = np.concatenate([
        np.zeros(len(Z_healthy_train), dtype="float32"),
        np.ones(len(Z_failed_train), dtype="float32"),
    ])

    Z_val_clf = np.concatenate([Z_healthy_val, Z_failed_val])
    y_val_clf = np.concatenate([
        np.zeros(len(Z_healthy_val), dtype="float32"),
        np.ones(len(Z_failed_val), dtype="float32"),
    ])

    #class weights
    weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train_clf)
    class_weight_dict = {0: float(weights[0]), 1: float(weights[1])}

    #treniranje Stage 2 klasifikatorja
    classifier = build_classifier(input_dim=dim)
    classifier.fit(
        Z_train_clf, y_train_clf,
        validation_data=(Z_val_clf, y_val_clf),
        epochs=60,
        batch_size=64,
        class_weight=class_weight_dict,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_auc", patience=10, mode="max", restore_best_weights=True
            )
        ],
        verbose=0,
    )

    #evalvacija
    y_prob = classifier.predict(Z_val_clf, batch_size=4096, verbose=0).flatten()
    best_threshold = find_best_threshold(y_val_clf, y_prob)
    y_pred = (y_prob >= best_threshold).astype(int)

    roc_auc = float(roc_auc_score(y_val_clf, y_prob))
    pr_auc = float(average_precision_score(y_val_clf, y_prob))

    failure_mask = y_val_clf == 1
    failure_recall = float(y_pred[failure_mask].mean()) if failure_mask.sum() > 0 else 0.0

    pred_positive_mask = y_pred == 1
    failure_precision = (
        float(y_val_clf[pred_positive_mask].mean())
        if pred_positive_mask.sum() > 0 else 0.0
    )

    f1 = (
        2 * failure_precision * failure_recall / (failure_precision + failure_recall + 1e-8)
    )

    return {
        "bottleneck_dim": dim,
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "failure_recall": round(failure_recall, 4),
        "failure_precision": round(failure_precision, 4),
        "failure_f1": round(f1, 4),
        "threshold": round(best_threshold, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep bottleneck dimenzij za iskanje optimalne za Impl 2."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(PROJECT_ROOT / "DiskData"),
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=100,
        help="Omejitev CSV datotek za hitrejši sweep (default: 100). Za polni sweep: None.",
    )
    parser.add_argument("--healthy-per-file", type=int, default=300)
    parser.add_argument("--failure-per-file", type=int, default=50)
    parser.add_argument(
        "--autoencoder-epochs",
        type=int,
        default=30,
        help="Maks. epohe za vsak avtoenkoder v sweepu (early stopping bo ustavil prej).",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--dims",
        type=int,
        nargs="+",
        default=DEFAULT_DIMS,
        help="Dimenzije za testiranje, npr. --dims 4 6 7 8 10 12",
    )
    args = parser.parse_args()

    np.random.seed(args.random_state)
    random.seed(args.random_state)
    tf.random.set_seed(args.random_state)

    #podatki se naložijo enkrat in so skupni za vse dimenzije (poštena primerjava)
    print(f"Gradim dataset (max_files={args.max_files})...")
    healthy_raw, failure_raw = build_dataset_from_many_csvs(
        data_dir=Path(args.data_dir),
        max_files=args.max_files,
        healthy_per_file=args.healthy_per_file,
        failure_per_file=args.failure_per_file,
        random_state=args.random_state,
    )

    if failure_raw.empty:
        raise RuntimeError("Ni najdenih failure vrstic. Povečaj --failure-per-file.")

    print(f"Healthy: {len(healthy_raw):,} | Failure: {len(failure_raw):,}")

    X_healthy = prepare_features(healthy_raw)
    X_failed = prepare_features(failure_raw)

    X_train_raw, X_val_raw = train_test_split(
        X_healthy, test_size=0.2, random_state=args.random_state
    )

    #isti scaler za vse dime — poštena primerjava
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw).astype("float32")
    X_val_scaled = scaler.transform(X_val_raw).astype("float32")
    X_failed_scaled = scaler.transform(X_failed).astype("float32")

    results = []

    for dim in args.dims:
        print(f"\n{'='*55}")
        print(f"  bottleneck_dim = {dim}  (avtoenkoder + klasifikator)")
        print(f"{'='*55}")

        result = train_and_evaluate_dim(
            dim=dim,
            X_train_scaled=X_train_scaled,
            X_val_scaled=X_val_scaled,
            X_failed_scaled=X_failed_scaled,
            ae_epochs=args.autoencoder_epochs,
            random_state=args.random_state,
        )
        results.append(result)
        print(
            f"  ROC-AUC: {result['roc_auc']:.4f} | "
            f"PR-AUC: {result['pr_auc']:.4f} | "
            f"Recall: {result['failure_recall']:.4f} | "
            f"F1: {result['failure_f1']:.4f}"
        )

    #razvrščeni po ROC-AUC
    results_sorted = sorted(results, key=lambda r: r["roc_auc"], reverse=True)
    best = results_sorted[0]

    print(f"\n{'='*55}")
    print("REZULTATI SWEPA (razvrščeni po ROC-AUC):\n")
    header = f"{'Dim':>5} | {'ROC-AUC':>8} | {'PR-AUC':>8} | {'Recall':>7} | {'Precis.':>8} | {'F1':>7}"
    print(header)
    print("-" * len(header))
    for r in results_sorted:
        marker = " ★" if r["bottleneck_dim"] == best["bottleneck_dim"] else ""
        print(
            f"{r['bottleneck_dim']:>5} | "
            f"{r['roc_auc']:>8.4f} | "
            f"{r['pr_auc']:>8.4f} | "
            f"{r['failure_recall']:>7.4f} | "
            f"{r['failure_precision']:>8.4f} | "
            f"{r['failure_f1']:>7.4f}"
            f"{marker}"
        )

    print(f"\n★  Priporočena bottleneck_dim: {best['bottleneck_dim']} (ROC-AUC: {best['roc_auc']:.4f})")
    print(
        f"   Za polni trening zaženi:\n"
        f"   python srcML/tensorflow_anomaly/train_autoencoder.py "
        f"--data-dir DiskData --bottleneck-dim {best['bottleneck_dim']}\n"
        f"   python srcML/tensorflow_classification/train_bottleneck_classifier.py "
        f"--data-dir DiskData"
    )

    sweep_output = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "experiment_config": {
            "max_files": args.max_files,
            "healthy_per_file": args.healthy_per_file,
            "failure_per_file": args.failure_per_file,
            "autoencoder_epochs": args.autoencoder_epochs,
            "dims_tested": args.dims,
        },
        "results": results_sorted,
        "recommended_bottleneck_dim": best["bottleneck_dim"],
    }

    DISKJSON_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DISKJSON_DIR / "bottleneck_sweep_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sweep_output, f, indent=2, ensure_ascii=False)

    print(f"\nRezultati shranjeni: {output_path}")


if __name__ == "__main__":
    main()
