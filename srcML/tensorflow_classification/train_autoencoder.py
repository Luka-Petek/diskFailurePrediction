#python srcML/tensorflow_classification/train_autoencoder.py --data-dir DiskData

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from srcML.nn_preprocessing.preprocessing import (
    FEATURE_COLUMNS,
    build_dataset_from_many_csvs,
    prepare_features,
    reconstruction_errors,
)

#artefakti Impl 2 avtoenkoder — loceni od Impl 1 (tensorflow_anomaly/)
CLF_AUTOENCODER_PATH = OUTPUT_DIR / "disk_clf_autoencoder.keras"
CLF_ENCODER_PATH = OUTPUT_DIR / "disk_clf_encoder.keras"
CLF_SCALER_PATH = OUTPUT_DIR / "clf_scaler.pkl"
CLF_AE_METADATA_PATH = OUTPUT_DIR / "clf_ae_metadata.json"

#arhitektura in nacrt mreze (komentarji so moja interpretacija in IZRAZITO AMATERSKI :) )
def build_autoencoder(input_dim: int, bottleneck_dim: int = 12) -> tf.keras.Model:

    #vhodni layer, vektor dolzine, npr. 19 nevronov
    inputs = tf.keras.Input(shape=(input_dim,), name="smart_features")

    #linearna tranformacija z = Wx + b, ki ji sledi aktivacijska funkcija ReLU
    #tukaj se mreza nauci kompleksne nelinerne povezave med atributi
    x = tf.keras.layers.Dense(64, activation="relu")(inputs)

    #normaliziranje aktivacij znotraj paketa, preprecuje "Vanishing Gradient" ??
    x = tf.keras.layers.BatchNormalization()(x)

    #zaradi overfittinga med vsakim korakom učenja nakljucno postavi 10% nevronov na 0
    x = tf.keras.layers.Dropout(0.10)(x)

    #dodatno nelinearno stiskanje informacij
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)

    #bottleneck da odstranimo šum, pomembne informacije stisnjene v bottleneck_dim nevronov
    bottleneck = tf.keras.layers.Dense(bottleneck_dim, activation="relu", name="bottleneck")(x)

    #rekonstrukcija prvotnih 12-dimenzionalnega prostora
    x = tf.keras.layers.Dense(32, activation="relu")(bottleneck)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Dense(64, activation="relu")(x)
    #številko nevronov se spet ujema z vhodnimi (19)
    outputs = tf.keras.layers.Dense(input_dim, activation="sigmoid", name="reconstruction")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="disk_smart_autoencoder")

    #tukaj dolocimo OPTIMIZATOR, tuki je Adam:
    #ne uporablja fiksne stopnje ucenja za vse parametre, ampak se prilagaja za vsako utez posebej
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mae", #mae formula za izračun napake
        metrics=["mse"],
    )

    return model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Treniranje Impl 2 avtoenkoder (locenega od Impl 1) za SMART anomaly detection."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(PROJECT_ROOT / "DiskData"),
        help="Mapa z Backblaze/SMART CSV datotekami.",
    )
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--healthy-per-file", type=int, default=1000)
    parser.add_argument("--failure-per-file", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--threshold-percentile", type=float, default=99.0)
    #za klasifikacijo je manjsi bottleneck ponavadi boljsi (sweep bo potrdil)
    parser.add_argument("--bottleneck-dim", type=int, default=8)
    args = parser.parse_args()

    np.random.seed(args.random_state)
    random.seed(args.random_state)
    tf.random.set_seed(args.random_state)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Gradim dataset iz CSV datotek...")
    healthy_raw, failure_raw = build_dataset_from_many_csvs(
        data_dir=Path(args.data_dir),
        max_files=args.max_files,
        healthy_per_file=args.healthy_per_file,
        failure_per_file=args.failure_per_file,
        random_state=args.random_state,
    )

    print(f"Healthy raw rows: {len(healthy_raw):,}")
    print(f"Failure eval raw rows: {len(failure_raw):,}")

    X_healthy = prepare_features(healthy_raw)

    X_train, X_val = train_test_split(
        X_healthy,
        test_size=0.2,
        random_state=args.random_state,
        shuffle=True,
    )

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype("float32")
    X_val_scaled = scaler.transform(X_val).astype("float32")

    model = build_autoencoder(input_dim=X_train_scaled.shape[1], bottleneck_dim=args.bottleneck_dim)
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
        ),
    ]

    print("\nZacenjam ucenje Impl 2 avtoenkoder...")
    history = model.fit(
        X_train_scaled,
        X_train_scaled,
        validation_data=(X_val_scaled, X_val_scaled),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        shuffle=True,
        verbose=1,
    )

    val_errors = reconstruction_errors(model, X_val_scaled)
    threshold = float(np.percentile(val_errors, args.threshold_percentile))
    p95 = float(np.percentile(val_errors, 95))
    p99 = float(np.percentile(val_errors, 99))
    p999 = float(np.percentile(val_errors, 99.9))
    mean_error = float(np.mean(val_errors))
    std_error = float(np.std(val_errors))

    evaluation = {}
    if not failure_raw.empty:
        X_failure = prepare_features(failure_raw)
        X_failure_scaled = scaler.transform(X_failure).astype("float32")
        failure_errors = reconstruction_errors(model, X_failure_scaled)
        y_true = np.concatenate([
            np.zeros_like(val_errors, dtype=int),
            np.ones_like(failure_errors, dtype=int),
        ])
        y_score = np.concatenate([val_errors, failure_errors])
        y_pred = (y_score > threshold).astype(int)
        evaluation = {
            "roc_auc": float(roc_auc_score(y_true, y_score)),
            "pr_auc": float(average_precision_score(y_true, y_score)),
            "validation_healthy_anomaly_rate": float(np.mean(val_errors > threshold)),
            "failure_eval_anomaly_rate": float(np.mean(failure_errors > threshold)),
            "classification_report": classification_report(
                y_true, y_pred,
                target_names=["healthy", "failure"],
                output_dict=True,
                zero_division=0,
            ),
        }
        print(f"\nROC-AUC: {evaluation['roc_auc']:.4f}")
        print(f"PR-AUC:  {evaluation['pr_auc']:.4f}")
        print(f"Failure anomaly rate: {evaluation['failure_eval_anomaly_rate']:.4f}")

    # !! REZULTATI TU NISO POMEMBNI, KER SO SAM MANJŠI VZOREC ZNACILNIC, KI GA BO KLASIFIKATOR UPORABIL KASNEJE
    metadata = {
        "model_type": "dense_autoencoder_clf",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "feature_columns": FEATURE_COLUMNS,
        "bottleneck_dim": args.bottleneck_dim,
        "threshold_percentile": args.threshold_percentile,
        "threshold": threshold,
        "validation_error_mean": mean_error,
        "validation_error_std": std_error,
        "validation_error_p95": p95,
        "validation_error_p99": p99,
        "validation_error_p999": p999,
        "normalization": {
            "score_formula": "(error - threshold) / (p999 - threshold), clipped to 0..1",
            "score_p999": p999,
        },
        "training": {
            "healthy_rows": int(len(X_train)),
            "validation_rows": int(len(X_val)),
            "epochs_requested": args.epochs,
            "epochs_finished": int(len(history.history["loss"])),
            "batch_size": args.batch_size,
        },
        "evaluation": evaluation,
    }

    model.save(CLF_AUTOENCODER_PATH)

    #izvlecemo encoder submodel — ta gre kot vhod v Stage 2 klasifikator
    encoder = tf.keras.Model(
        inputs=model.input,
        outputs=model.get_layer("bottleneck").output,
        name="disk_clf_encoder",
    )
    encoder.save(CLF_ENCODER_PATH)

    joblib.dump(scaler, CLF_SCALER_PATH)

    with open(CLF_AE_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("\nShranjeno (Impl 2 avtoenkoder):")
    print(f"Autoencoder: {CLF_AUTOENCODER_PATH}")
    print(f"Encoder:     {CLF_ENCODER_PATH}")
    print(f"Scaler:      {CLF_SCALER_PATH}")
    print(f"Metadata:    {CLF_AE_METADATA_PATH}")
    print(f"Bottleneck dim: {args.bottleneck_dim}")


if __name__ == "__main__":
    main()