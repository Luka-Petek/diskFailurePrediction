# Thin alias module to keep naming consistent after HIR→AHI rename.
# Re-exports predict_ahi CLI from hir_final without duplicating logic.

from __future__ import annotations

import argparse
from pathlib import Path
import json
import sys

from srcML.hir_final import predict_ahi as _predict_ahi, PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AHI — kombinirana napoved zdravja diska (4 modeli).",
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

    result = _predict_ahi(
        smartctl_json_path=Path(args.input),
        sklearn_dir=Path(args.sklearn_dir),
        clf_dir=Path(args.clf_dir),
        anomaly_dir=Path(args.anomaly_dir),
        clustering_dir=Path(args.clustering_dir),
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
