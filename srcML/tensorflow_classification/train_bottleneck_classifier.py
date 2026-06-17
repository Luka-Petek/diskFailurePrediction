#python srcML/tensorflow_classification/train_bottleneck_classifier.py --data-dir DiskData

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from srcML.nn_preprocessing.preprocessing import (
    FEATURE_COLUMNS,
    build_balanced_dataset_from_csvs,
    prepare_features,
)

DEFAULT_FAILURE_CSV = PROJECT_ROOT / "csv" / "vseOdpovedi.csv"

#artefakti Impl 2 avtoenkoder (iz tensorflow_classification/train_autoencoder.py)
ENCODER_PATH = OUTPUT_DIR / "disk_clf_encoder.keras"
SCALER_PATH = OUTPUT_DIR / "clf_scaler.pkl"
AE_METADATA_PATH = OUTPUT_DIR / "clf_ae_metadata.json"

CLASSIFIER_PATH = OUTPUT_DIR / "disk_bottleneck_classifier.keras"
METADATA_PATH = OUTPUT_DIR / "bottleneck_metadata.json"


def build_classifier(input_dim: int) -> tf.keras.Model:
    #vhod: bottleneck features (6-12 dimenzij)
    inputs = tf.keras.Input(shape=(input_dim,), name="bottleneck_features")

    #manjša gosta mreža — bottleneck features so že čiste, ne potrebujemo globoke arhitekture
    x = tf.keras.layers.Dense(16, activation="relu")(inputs)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(8, activation="relu")(x)

    #sigmoid izhod → P(failure) ∈ [0, 1]
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="failure_prob")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="bottleneck_failure_classifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def extract_bottleneck_features(encoder: tf.keras.Model, X_scaled: np.ndarray) -> np.ndarray:
    return encoder.predict(X_scaled, batch_size=4096, verbose=0).astype("float32")


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    #poiščemo prag, ki maksimizira F1 na validacijski množici
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    best_idx = int(np.argmax(f1_scores[:-1]))
    return float(thresholds[best_idx])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Treniranje Stage 2 klasifikatorja na bottleneck features (Impl 2)."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(PROJECT_ROOT / "DiskData"),
        help="Mapa z Backblaze/SMART CSV datotekami.",
    )
    parser.add_argument("--max-failure", type=int, default=None,
                        help="Maks. stevilo failure vrstic (default=vse razpolozljive).")
    #ce obstaja, preskoce Pass 1 skeniranje in nalozi failure vrstice direktno
    parser.add_argument("--failure-csv", type=str,
                        default=str(DEFAULT_FAILURE_CSV) if DEFAULT_FAILURE_CSV.exists() else None,
                        help="Pot do CSV z vsemi failure vrsticami (preskoce Pass 1).")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.random_state)
    random.seed(args.random_state)
    tf.random.set_seed(args.random_state)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    #nalaganje Impl 2 avtoenkoder artefaktov (zamrznjen encoder — ne treniramo ga vec)
    print("Nalagam Impl 2 encoder in scaler...")
    encoder = tf.keras.models.load_model(ENCODER_PATH)
    encoder.trainable = False

    scaler = joblib.load(SCALER_PATH)

    with open(AE_METADATA_PATH, encoding="utf-8") as f:
        ae_metadata = json.load(f)
    bottleneck_dim = ae_metadata["bottleneck_dim"]
    print(f"Bottleneck dim (iz Impl 2 clf_ae_metadata.json): {bottleneck_dim}")

    #gradnja 50:50 dataseta — zbere VSE failure vrstice, izenaci z zdravimi
    print("\nGradim uravnotezen dataset iz CSV datotek...")
    failure_csv = Path(args.failure_csv) if args.failure_csv else None
    healthy_raw, failure_raw = build_balanced_dataset_from_csvs(
        data_dir=Path(args.data_dir),
        max_failure=args.max_failure,
        random_state=args.random_state,
        failure_csv=failure_csv,
    )

    if failure_raw.empty:
        raise RuntimeError("Ni najdenih failure vrstic — potrebnih za supervised trening Impl 2!")

    print(f"Healthy raw rows: {len(healthy_raw):,}")
    print(f"Failure raw rows: {len(failure_raw):,}")

    #preprocessing in skaliranje (isti scaler kot Impl 1)
    X_healthy = prepare_features(healthy_raw)
    X_failed = prepare_features(failure_raw)

    X_healthy_scaled = scaler.transform(X_healthy).astype("float32")
    X_failed_scaled = scaler.transform(X_failed).astype("float32")

    #ekstrakcija bottleneck features — encoder pretvori 19 → N čistih dimenzij
    print("\nEkstrahiram bottleneck features skozi zamrznjeni encoder...")
    Z_healthy = extract_bottleneck_features(encoder, X_healthy_scaled)
    Z_failed = extract_bottleneck_features(encoder, X_failed_scaled)

    print(f"Bottleneck features shape — healthy: {Z_healthy.shape} | failed: {Z_failed.shape}")

    #sestavimo označeni dataset: healthy=0, failed=1
    Z = np.concatenate([Z_healthy, Z_failed])
    y = np.concatenate([
        np.zeros(len(Z_healthy), dtype="float32"),
        np.ones(len(Z_failed), dtype="float32"),
    ])

    #razdelitev: 70% train, 15% val, 15% test — stratified da ohranimo razmerje razredov
    Z_train, Z_temp, y_train, y_temp = train_test_split(
        Z, y, test_size=0.30, random_state=args.random_state, stratify=y
    )
    Z_val, Z_test, y_val, y_test = train_test_split(
        Z_temp, y_temp, test_size=0.50, random_state=args.random_state, stratify=y_temp
    )

    print(f"\nRazdelitev:")
    print(f"  Train: {len(Z_train):,} (failures: {int(y_train.sum()):,})")
    print(f"  Val:   {len(Z_val):,}   (failures: {int(y_val.sum()):,})")
    print(f"  Test:  {len(Z_test):,}  (failures: {int(y_test.sum()):,})")

    #class weights za uravnoteženje — okvarjeni diski so redki (manjšina)
    weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)
    class_weight_dict = {0: float(weights[0]), 1: float(weights[1])}
    print(f"\nClass weights: healthy={class_weight_dict[0]:.2f} | failure={class_weight_dict[1]:.2f}")

    #gradnja Stage 2 klasifikatorja
    classifier = build_classifier(input_dim=bottleneck_dim)
    classifier.summary()

    #TensorBoard logi
    log_dir = OUTPUT_DIR / "logs" / datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_cb = tf.keras.callbacks.TensorBoard(
        log_dir=str(log_dir),
        histogram_freq=1,
        write_graph=False,
        update_freq="epoch",
    )

    #rocno zapisi graf klasifikatorja (write_graph=True v callbacku ne dela z Keras)
    writer = tf.summary.create_file_writer(str(log_dir))
    tf.summary.trace_on(graph=True, profiler=False)
    classifier(tf.zeros([1, bottleneck_dim]), training=False)
    with writer.as_default():
        #tensorboard ne dela dobro z .keras, zapises graf rocno
        tf.summary.trace_export(name="classifier_graph", step=0)
    writer.flush()

    #EarlyStopping gleda na AUC (bolj relevantno kot loss pri imbalanced podatkih)
    callbacks = [
        tensorboard_cb,
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            patience=15,
            mode="max",
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc",
            factor=0.5,
            patience=5,
            mode="max",
            min_lr=1e-6,
        ),
    ]

    print("\nZačenjam učenje Stage 2 klasifikatorja...")
    history = classifier.fit(
        Z_train, y_train,
        validation_data=(Z_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1,
    )

    #iskanje optimalnega praga na validacijski množici (maksimizira F1)
    y_val_prob = classifier.predict(Z_val, batch_size=4096, verbose=0).flatten()
    best_threshold = find_best_threshold(y_val, y_val_prob)
    print(f"\nOptimalen prag (F1 na val množici): {best_threshold:.4f}")

    #končna evalvacija na testni množici (ta del podatkov model ni videl)
    y_test_prob = classifier.predict(Z_test, batch_size=4096, verbose=0).flatten()
    y_test_pred = (y_test_prob >= best_threshold).astype(int)

    roc_auc = float(roc_auc_score(y_test, y_test_prob))
    pr_auc = float(average_precision_score(y_test, y_test_prob))
    clf_report = classification_report(
        y_test, y_test_pred,
        target_names=["healthy", "failure"],
        output_dict=True,
        zero_division=0,
    )

    print(f"\nEvalvacija na testni množici:")
    print(f"ROC-AUC:          {roc_auc:.4f}")
    print(f"PR-AUC:           {pr_auc:.4f}")
    print(f"Failure recall:   {clf_report['failure']['recall']:.4f}")
    print(f"Failure precision:{clf_report['failure']['precision']:.4f}")
    print(f"Failure F1:       {clf_report['failure']['f1-score']:.4f}")

    metadata = {
        "model_type": "bottleneck_classifier",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "feature_columns": FEATURE_COLUMNS,
        "bottleneck_dim": bottleneck_dim,
        "encoder_source": str(ENCODER_PATH),
        "threshold": best_threshold,
        "training": {
            "healthy_rows": int(len(Z_healthy)),
            "failure_rows": int(len(Z_failed)),
            "train_rows": int(len(Z_train)),
            "val_rows": int(len(Z_val)),
            "test_rows": int(len(Z_test)),
            "epochs_requested": args.epochs,
            "epochs_finished": int(len(history.history["loss"])),
            "batch_size": args.batch_size,
            "class_weight_healthy": class_weight_dict[0],
            "class_weight_failure": class_weight_dict[1],
        },
        "evaluation": {
            "roc_auc": roc_auc, # sensitivity (recall za pozitiven razred) vs specificity (recall za negatoven razred)
            "pr_auc": pr_auc,   # sensitivity (recall) vs precision
            "failure_recall": clf_report["failure"]["recall"],
            "failure_precision": clf_report["failure"]["precision"],
            "failure_f1": clf_report["failure"]["f1-score"],
            "classification_report": clf_report,
        },
    }

    classifier.save(CLASSIFIER_PATH)

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("\nShranjeno:")
    print(f"Classifier: {CLASSIFIER_PATH}")
    print(f"Metadata:   {METADATA_PATH}")
    print(f"Threshold:  {best_threshold:.4f}")
    print(f"\nTensorBoard logs: {log_dir}")
    print(f"Zaženi: tensorboard --logdir \"{OUTPUT_DIR / 'logs'}\"")


if __name__ == "__main__":
    main()
