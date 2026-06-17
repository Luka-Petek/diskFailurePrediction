#   python srcML/tensorflow_anomaly/predict_autoencoder.py --input DiskJson/disk_data_sda.json

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from srcML.sklearn.disk_pipeline import pretvori_json_v_surovi_df
from srcML.tensorflow_anomaly.train_autoencoder import prepare_features


def load_artifacts(model_dir: Path) -> tuple:
    model = tf.keras.models.load_model(model_dir / "disk_autoencoder.keras")
    scaler = joblib.load(model_dir / "tf_scaler.pkl")
    with open(model_dir / "tf_metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)
    return model, scaler, metadata


def predict(smartctl_json_path: Path, model_dir: Path) -> dict:
    model, scaler, metadata = load_artifacts(model_dir)

    with open(smartctl_json_path, encoding="utf-8") as f:
        smartctl_dict = json.load(f)

    raw_df = pretvori_json_v_surovi_df(smartctl_dict)
    X = prepare_features(raw_df)
    X_scaled = scaler.transform(X).astype("float32")

    reconstructed = model.predict(X_scaled, batch_size=1, verbose=0)
    reconstruction_error = float(np.mean(np.abs(X_scaled - reconstructed)))

    threshold = metadata["threshold"]
    p999 = metadata["normalization"]["score_p999"]

    anomaly = reconstruction_error > threshold

    #normaliziran score 0..1 (0 = zdravo, 1 = mocna anomalija)
    if p999 <= threshold:
        anomaly_score = 0.0
    else:
        anomaly_score = float(np.clip(
            (reconstruction_error - threshold) / (p999 - threshold), 0.0, 1.0
        ))

    return {
        "anomaly": bool(anomaly),
        "anomaly_score": round(anomaly_score, 4),
        "reconstruction_error": round(reconstruction_error, 6),
        "threshold": round(threshold, 6),
        "verdict": "ANOMALY_DETECTED" if anomaly else "HEALTHY",
        "model_info": {
            "bottleneck_dim": metadata.get("bottleneck_dim", 12),
            "threshold_percentile": metadata["threshold_percentile"],
            "trained_on_healthy_rows": metadata["training"]["healthy_rows"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Napoved anomalije za posamezen disk z avtoenkoderjem (Impl 1)."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Pot do smartctl JSON datoteke (smartctl -A -i -j).",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(OUTPUT_DIR),
        help="Mapa z disk_autoencoder.keras, tf_scaler.pkl in tf_metadata.json.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Napaka: datoteka ne obstaja: {input_path}", file=sys.stderr)
        sys.exit(1)

    result = predict(
        smartctl_json_path=input_path,
        model_dir=Path(args.model_dir),
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
