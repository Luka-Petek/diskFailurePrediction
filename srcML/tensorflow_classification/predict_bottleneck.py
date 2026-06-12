#python srcML/tensorflow_classification/predict_bottleneck.py --input DiskJson/disk_data_sda.json

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

from srcML.disk_pipeline import pretvori_json_v_surovi_df
from srcML.nn_preprocessing.preprocessing import prepare_features


def load_artifacts(clf_dir: Path) -> tuple:
    #vsi artefakti Impl 2 so v tensorflow_classification/
    encoder = tf.keras.models.load_model(clf_dir / "disk_clf_encoder.keras")
    classifier = tf.keras.models.load_model(clf_dir / "disk_bottleneck_classifier.keras")
    scaler = joblib.load(clf_dir / "clf_scaler.pkl")
    with open(clf_dir / "bottleneck_metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)
    return encoder, classifier, scaler, metadata


def predict(smartctl_json_path: Path, clf_dir: Path) -> dict:
    encoder, classifier, scaler, metadata = load_artifacts(clf_dir)

    with open(smartctl_json_path, encoding="utf-8") as f:
        smartctl_dict = json.load(f)

    raw_df = pretvori_json_v_surovi_df(smartctl_dict)
    X = prepare_features(raw_df)
    X_scaled = scaler.transform(X).astype("float32")

    #Stage 1: encoder → bottleneck features (19 → N čistih dimenzij)
    bottleneck_features = encoder.predict(X_scaled, batch_size=1, verbose=0)

    #Stage 2: klasifikator → P(failure)
    failure_prob = float(
        classifier.predict(bottleneck_features, batch_size=1, verbose=0).flatten()[0]
    )

    threshold = metadata["threshold"]
    anomaly = failure_prob >= threshold

    return {
        "failure_predicted": bool(anomaly),
        "failure_probability": round(failure_prob, 4),
        "threshold": round(threshold, 4),
        "verdict": "FAILURE" if anomaly else "HEALTHY",
        "bottleneck_features": [round(float(v), 4) for v in bottleneck_features.flatten()],
        "model_info": {
            "bottleneck_dim": metadata["bottleneck_dim"],
            "trained_on_failure_rows": metadata["training"]["failure_rows"],
            "failure_recall": metadata["evaluation"]["failure_recall"],
            "failure_f1": metadata["evaluation"]["failure_f1"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Napoved anomalije za posamezen disk z bottleneck klasifikatorjem (Impl 2)."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Pot do smartctl JSON datoteke (smartctl -A -i -j).",
    )
    parser.add_argument(
        "--clf-dir",
        type=str,
        default=str(OUTPUT_DIR),
        help="Mapa z vsemi Impl 2 artefakti (disk_clf_encoder.keras, clf_scaler.pkl, bottleneck_metadata.json).",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Napaka: datoteka ne obstaja: {input_path}", file=sys.stderr)
        sys.exit(1)

    result = predict(
        smartctl_json_path=input_path,
        clf_dir=Path(args.clf_dir),
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
