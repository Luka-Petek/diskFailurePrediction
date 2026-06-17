# srcML/tensorflow_anomaly/train_autoencoder.py

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
SRCML_ROOT = PROJECT_ROOT / "srcML"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

#uporabimo procesiranje od skleanr
from srcML.sklearn.disk_pipeline import procesiraj_podatke
#preprocessing za NN
from srcML.nn_preprocessing.preprocessing import build_dataset_from_many_csvs, prepare_features, reconstruction_errors

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
#struktura in utezi nevronske mreze
MODEL_PATH = OUTPUT_DIR / "disk_autoencoder.keras"
ENCODER_PATH = OUTPUT_DIR / "disk_encoder.keras"
SCALER_PATH = OUTPUT_DIR / "tf_scaler.pkl"
#trenshold in rezultati
METADATA_PATH = OUTPUT_DIR / "tf_metadata.json"

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
        description="Treniranje TensorFlow autoencoderja za SMART anomaly detection."
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(PROJECT_ROOT / "DiskData"),
        help="Mapa z velikimi Backblaze/SMART CSV datotekami.",
    )
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--healthy-per-file", type=int, default=1000)
    parser.add_argument("--failure-per-file", type=int, default=100)
    #kolikokrat se nevronska mreza sprehodi cez datasat (in sproti popravlja utezi)
    parser.add_argument("--epochs", type=int, default=60)
    #koliko vrstic se pogledat hkrati — 128 je eksperimentalno najboljsi (glej DiskJson/)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--random-state", type=int, default=42)
    #meja, kjer se disk smatra za anomalijo... višji %, manjša občutljivost na anomalije
    parser.add_argument("--threshold-percentile", type=float, default=99.0)
    #stevilo nevronov v bottleneck sloju — impl 1 uporablja 12, impl 2 bo eksperimentirala
    parser.add_argument("--bottleneck-dim", type=int, default=12)

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
    model = build_autoencoder(input_dim=X_train_scaled.shape[1], bottleneck_dim=args.bottleneck_dim)
    model.summary()

    #shranjevanje log-ov za tensorbaord
    log_dir = OUTPUT_DIR / "logs" / ("fit_" + datetime.now().strftime("%Y%m%d-%H%M%S"))

    #tensorbaord prikazi
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True,
    )

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
        tensorboard_callback
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

    #stevilo okvarjenih diskov ki jih podamo preko parametra so ZA TESTNO MNOZICO !, tukaj:
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
        #sposobnost modela, da loči med zdravimi in okvarjenimi diski (1.0 je idealno)
        print(f"ROC-AUC: {evaluation['roc_auc']:.4f}")

        #uspešnost iskanja redkih okvar brez povzročanja lažnih alarmov (bolj realna ocena)
        print(f"PR-AUC:  {evaluation['pr_auc']:.4f}")

        #stopnja lažnih alarmov (delež zdravih diskov, ki so bili napačno označeni kot anomalija)
        print(
            f"Healthy anomaly rate: {evaluation['validation_healthy_anomaly_rate']:.4f}"
        )

        #recall / Občutljivost (delež dejansko okvarjenih diskov, ki jih je model uspešno ujel)
        print(
            f"Failure anomaly rate: {evaluation['failure_eval_anomaly_rate']:.4f}"
        )

    #izpis diskov?
    metadata = {
        "model_type": "dense_autoencoder",
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

    model.save(MODEL_PATH)

    #izvlecemo encoder submodel (input → bottleneck), ki ga impl 2 uporabi kot feature extractor
    encoder = tf.keras.Model(
        inputs=model.input,
        outputs=model.get_layer("bottleneck").output,
        name="disk_encoder",
    )
    encoder.save(ENCODER_PATH)

    joblib.dump(scaler, SCALER_PATH)

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("\nShranjeno:")
    print(f"Model:    {MODEL_PATH}")
    print(f"Encoder:  {ENCODER_PATH}")
    print(f"Scaler:   {SCALER_PATH}")
    print(f"Metadata: {METADATA_PATH}")
    print(f"Threshold: {threshold:.6f}")


if __name__ == "__main__":
    main()

#VARIACIJE PARAMETROV (iscem najbolse):

#python srcML/tensorflow_anomaly/train_autoencoder.py --data-dir DiskData --healthy-per-file 1000 --failure-per-file 100 --epochs 60 --batch-size 512

#python srcML/tensorflow_anomaly/train_autoencoder.py --data-dir DiskData --healthy-per-file 1000 --failure-per-file 100 --epochs 60 --batch-size  --> 01_nn_results.json