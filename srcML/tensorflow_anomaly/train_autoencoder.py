# srcML/tensorflow_anomaly/train_autoencoder.py

import argparse
import glob
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRCML_ROOT = PROJECT_ROOT / "srcML"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from srcML.disk_pipeline import procesiraj_podatke


FEATURE_COLUMNS = [
    "capacity_gigabytes",
    "jeSSD",
    "smart_1_raw",
    "smart_3_raw",
    "smart_4_raw",
    "smart_5_raw",
    "smart_7_raw",
    "smart_9_raw",
    "smart_12_raw",
    "smart_187_raw",
    "smart_188_raw",
    "smart_191_raw",
    "smart_192_raw",
    "smart_193_raw",
    "smart_197_raw",
    "smart_198_raw",
    "any_critical_error",
    "total_error_count",
    "error_per_gb",
]


OUTPUT_DIR = SRCML_ROOT / "tensorflow_anomaly"
MODEL_PATH = OUTPUT_DIR / "disk_autoencoder.keras"
SCALER_PATH = OUTPUT_DIR / "tf_scaler.pkl"
METADATA_PATH = OUTPUT_DIR / "tf_metadata.json"


def read_csv_robust(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        #ce ne dela privzeto ločilo
        return pd.read_csv(path, sep=";", low_memory=False)

#iz posameznega CSV-ja definiramo koliko instanc hocemo in katerih
def sample_rows_from_csv(csv_path: str, healthy_per_file: int, failure_per_file: int, random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = read_csv_robust(csv_path)

    if "failure" not in df.columns:
        return pd.DataFrame(), pd.DataFrame()

    healthy = df[df["failure"] == 0]
    failed = df[df["failure"] == 1]

    healthy_sample = pd.DataFrame()
    failed_sample = pd.DataFrame()

    if not healthy.empty:
        n = min(len(healthy), healthy_per_file)
        healthy_sample = healthy.sample(n=n, random_state=random_state)

    if not failed.empty and failure_per_file > 0:
        n = min(len(failed), failure_per_file)
        failed_sample = failed.sample(n=n, random_state=random_state)

    return healthy_sample, failed_sample

#dejansko pobiranje in agregiranje datotek, spet glede na arg
def build_dataset_from_many_csvs(data_dir: Path, max_files: int | None, healthy_per_file: int, failure_per_file: int, random_state: int,) -> tuple[pd.DataFrame, pd.DataFrame]:
    #iskanje .csv tudi za podmape
    pattern = str(data_dir / "**" / "*.csv")
    csv_files = glob.glob(pattern, recursive=True)

    if not csv_files:
        raise FileNotFoundError(f"Ni najdenih CSV datotek v: {data_dir}")

    random.Random(random_state).shuffle(csv_files)

    if max_files is not None:
        csv_files = csv_files[:max_files]

    healthy_parts = []
    failure_parts = []

    #for loop za cez datoteke
    for i, csv_path in enumerate(csv_files, start=1):
        try:
            healthy_sample, failed_sample = sample_rows_from_csv(
                csv_path=csv_path,
                healthy_per_file=healthy_per_file,
                failure_per_file=failure_per_file,
                random_state=random_state + i,
            )

            if not healthy_sample.empty:
                healthy_parts.append(healthy_sample)

            if not failed_sample.empty:
                failure_parts.append(failed_sample)

            #vsakih 25 predelanih datotek izpisemo stanje
            if i % 25 == 0:
                healthy_count = sum(len(x) for x in healthy_parts)
                failure_count = sum(len(x) for x in failure_parts)
                print(
                    f"[{i}/{len(csv_files)}] Healthy: {healthy_count:,} | Failure eval: {failure_count:,}"
                )

        except Exception as exc:
            print(f"Preskočena datoteka {csv_path}: {exc}")

    if not healthy_parts:
        raise RuntimeError("Ni bilo najdenih healthy vrstic za trening.")

    #zdruzimo vse v en dataframe
    healthy_df = pd.concat(healthy_parts, ignore_index=True)
    failure_df = (
        pd.concat(failure_parts, ignore_index=True)
        if failure_parts
        else pd.DataFrame()
    )

    return healthy_df, failure_df

#za ciscenje in trans. podatkov
def prepare_features(df_raw: pd.DataFrame) -> pd.DataFrame:

    #najprej skozi pipeline, ki je ze definiran od prej za sklearn ml
    df_processed = procesiraj_podatke(df_raw)

    X = pd.DataFrame(index=df_processed.index)

    #cez fiksne feature columns, ce jih ni damo 0
    for col in FEATURE_COLUMNS:
        if col in df_processed.columns:
            X[col] = df_processed[col]
        else:
            X[col] = 0.0

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0.0)

    #vsi stolpci v stevilske tipe
    for col in FEATURE_COLUMNS:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0)

    return X.astype("float32")

#arhitektura in nacrt mreze (komentarji so moja interpretacija in IZRAZITO AMATERSKI :) )
def build_autoencoder(input_dim: int) -> tf.keras.Model:

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

    #bottleneck da odstranimo šum, v teh 12 nevronov so bolj "bistvene" informacije
    bottleneck = tf.keras.layers.Dense(12, activation="relu", name="bottleneck")(x)

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

def reconstruction_errors(model: tf.keras.Model, X_scaled: np.ndarray) -> np.ndarray:
    reconstructed = model.predict(X_scaled, batch_size=4096, verbose=0)
    errors = np.mean(np.abs(X_scaled - reconstructed), axis=1)
    return errors


def normalize_score(error: float, threshold: float, p999: float) -> float:
    if p999 <= threshold:
        return 0.0

    score = (error - threshold) / (p999 - threshold)
    return float(np.clip(score, 0.0, 1.0))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Treniranje TensorFlow autoencoderja za SMART anomaly detection."
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(PROJECT_ROOT / "DiskData"),
        help="Mapa z velikimi Backblaze/SMART CSV datotekami.",
    )
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--healthy-per-file", type=int, default=500)
    parser.add_argument("--failure-per-file", type=int, default=50)
    #kolikokrat se nevronska mreza sprehodi cez datasat (in sproti popravlja utezi)... 80 je sweet spot
    parser.add_argument("--epochs", type=int, default=80)
    #koliko vrstic se pogledat hkrati, keras avtomatsko zracuna MAE za vseh 1024 instanc hkrati.. 1024 sweet sport
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--random-state", type=int, default=42)
    #meja, kjer se disk smatra za anomalijo... višji %, manjša občutljivost na anomalije
    parser.add_argument("--threshold-percentile", type=float, default=99.0)

    args = parser.parse_args()

    np.random.seed(args.random_state)
    random.seed(args.random_state)
    tf.random.set_seed(args.random_state)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    #klicemo funkcijo od prej za grajenje dataseta
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

    #testna / učna množica
    X_train, X_val = train_test_split(
        X_healthy,
        test_size=0.2,
        random_state=args.random_state,
        shuffle=True,
    )

    #centriramo podatke glede na mediano
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype("float32")
    X_val_scaled = scaler.transform(X_val).astype("float32")

    #klicemo funkcijo za gradnjo modela, dim dolocena s stevilom stolpcev matrike
    model = build_autoencoder(input_dim=X_train_scaled.shape[1])
    model.summary()

    callbacks = [
        #neki za zgodnje ustavljanje ??
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
        ),
        #neki za nizanje stopnje ucenja?
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
        ),
    ]

    #dejansko učenje
    print("Začenjam učenje autoencoderja...")
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

    #MAE na validac. mnozici
    val_errors = reconstruction_errors(model, X_val_scaled)

    #neki percentili za prikaz napak
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

        y_true = np.concatenate(
            [
                np.zeros_like(val_errors, dtype=int),
                np.ones_like(failure_errors, dtype=int),
            ]
        )
        y_score = np.concatenate([val_errors, failure_errors])

        y_pred = (y_score > threshold).astype(int)

        evaluation = {
            "roc_auc": float(roc_auc_score(y_true, y_score)),
            "pr_auc": float(average_precision_score(y_true, y_score)),
            "validation_healthy_anomaly_rate": float(np.mean(val_errors > threshold)),
            "failure_eval_anomaly_rate": float(np.mean(failure_errors > threshold)),
            "classification_report": classification_report(
                y_true,
                y_pred,
                target_names=["healthy", "failure"],
                output_dict=True,
                zero_division=0,
            ),
        }

        print("\nEvalvacija proti failure vrsticam:")
        print(f"ROC-AUC: {evaluation['roc_auc']:.4f}")
        print(f"PR-AUC:  {evaluation['pr_auc']:.4f}")
        print(
            f"Healthy anomaly rate: {evaluation['validation_healthy_anomaly_rate']:.4f}"
        )
        print(
            f"Failure anomaly rate: {evaluation['failure_eval_anomaly_rate']:.4f}"
        )

    #izpis diskov?
    metadata = {
        "model_type": "dense_autoencoder",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "feature_columns": FEATURE_COLUMNS,
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

    model.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("\nShranjeno:")
    print(f"Model:    {MODEL_PATH}")
    print(f"Scaler:   {SCALER_PATH}")
    print(f"Metadata: {METADATA_PATH}")
    print(f"Threshold: {threshold:.6f}")


if __name__ == "__main__":
    main()

#VARIACIJE PARAMETROV (iscem najbolse):

#python srcML/tensorflow_anomaly/train_autoencoder.py --data-dir DiskData --healthy-per-file 1000 --failure-per-file 100 --epochs 60 --batch-size 512

#python srcML/tensorflow_anomaly/train_autoencoder.py --data-dir DiskData --healthy-per-file 1000 --failure-per-file 100 --epochs 60 --batch-size  --> 01_nn_results.json