# Disk Failure Prediction — Neural Network Implementation Plan

## Experimental History (Autoencoder Runs)

| Run | Healthy Rows | Batch | Epochs | ROC-AUC | PR-AUC | Failure Recall |
|-----|-------------|-------|--------|---------|--------|----------------|
| 01  | 146K        | 1024  | 67     | 0.861   | 0.590  | 25.4%          |
| 02  | 292K        | 512   | 39     | 0.861   | 0.554  | 43.4%          |
| 03  | 438K        | 2048  | 60     | 0.783   | 0.201  | 12.8% ← worst  |
| 04  | 292K        | 256   | 58     | 0.882   | 0.568  | 43.7%          |
| 05  | 292K        | 128   | 37     | 0.901   | 0.600  | 44.7% ← best   |

**Key finding:** 292K healthy rows + batch_size=128 is the sweet spot.
Large batch (2048) destroys performance. More data (438K) does not help past 292K.

---

## Overview

Two separate, self-contained implementations. Both skip sklearn pipeline (unchanged).
Backend and frontend are left for later — testing done via predict scripts only.

---

## Implementation 1 — Autoencoder Anomaly Detector

> Train strictly on healthy disks → MAE reconstruction error → threshold → `anomaly: true/false`

### Architecture (unchanged)
```
Input(19) → Dense(64, relu) → BN → Dropout(0.10) → Dense(32, relu) → BN
         → Bottleneck(12, relu) → Dense(32, relu) → BN → Dense(64, relu) → Output(19, sigmoid)
```

### What the current train_autoencoder.py already does correctly
- Trains on healthy rows only (failed rows used for post-training evaluation only)
- MAE threshold at 99th percentile of validation errors
- EarlyStopping + ReduceLROnPlateau + TensorBoard callbacks
- Saves model + scaler + metadata

### Changes to train_autoencoder.py
1. **Fix defaults to best-known config**
   - `--batch-size` default: `1024` → `128`
   - `--healthy-per-file` default: `500` → `750` (yields ~292K total across dataset)
   - Add `--bottleneck-dim` argument, default `12` (Impl 1 keeps 12, Impl 2 will override)
2. **Save encoder submodel** after training (needed by Impl 2, free to add now)
   ```python
   encoder = tf.keras.Model(
       inputs=model.input,
       outputs=model.get_layer("bottleneck").output,
       name="disk_encoder"
   )
   encoder.save(OUTPUT_DIR / "disk_encoder.keras")
   ```

### New file: predict_autoencoder.py
Single-disk inference script for Implementation 1.
- **Input:** path to a `smartctl -j` JSON file (same format as files in `DiskJson/`)
- **Loads:** `disk_autoencoder.keras` + `tf_scaler.pkl` + `tf_metadata.json`
- **Pipeline:** raw JSON → `pretvori_json_v_surovi_df` → `procesiraj_podatke` → `prepare_features` → scale → `model.predict` → compute MAE → compare to threshold
- **Output:**
  ```json
  {
    "anomaly": true,
    "anomaly_score": 0.73,
    "reconstruction_error": 0.031,
    "threshold": 0.00407,
    "verdict": "ANOMALY_DETECTED"
  }
  ```

### Impl 1 — File Structure
```
srcML/tensorflow_anomaly/
├── train_autoencoder.py      ← MODIFY (fix defaults, add bottleneck_dim arg, save encoder)
├── predict_autoencoder.py    ← NEW    (single-disk inference)
├── disk_autoencoder.keras    ← artifact (retrain with corrected defaults)
├── disk_encoder.keras        ← NEW artifact (encoder submodel, reused by Impl 2)
├── tf_scaler.pkl             ← artifact
└── tf_metadata.json          ← artifact
```

---

## Implementation 2 — Bottleneck → Supervised Classifier (Professor's Recommendation)

### Motivation
Implementation 1 is **unsupervised** — the model never sees failed disks during training.
The threshold is a heuristic on reconstruction error. Failure recall ceiling is ~45%.

The professor's recommendation: use the bottleneck as a **denoising feature extractor**.
Failed disks end up in a different region of the latent space. A supervised classifier can
explicitly learn this boundary — expected to significantly outperform reconstruction error alone.

### What to predict from the bottleneck?
**Binary P(failure)** — the most principled choice given our labeled data.
- We have `failure = 0/1` labels for all rows
- The bottleneck gives a clean, compressed, noise-free representation (6-8 features vs 19)
- A small feedforward NN learns the healthy/failed boundary in latent space
- Output: `P(failure) ∈ [0, 1]`, threshold tuned on the PR curve (F1-maximizing)

### Architecture
```
Stage 1 — Encoder (trained unsupervised on healthy only, FROZEN in Stage 2):
  Raw SMART (19) → [MinMaxScaler] → Dense(64) → BN → Dropout → Dense(32) → BN → Bottleneck(N)

Stage 2 — Classifier (trained supervised on healthy + failed):
  Bottleneck(N) → Dense(16, relu) → Dropout(0.2) → Dense(8, relu) → Dense(1, sigmoid)
                                                                           ↓
                                                                      P(failure)
```

### Bottleneck Dimension
The professor suggests ~6-7 features (down from current 12).
Run `bottleneck_experiment.py` to confirm empirically before final training.
Sweep: `[4, 6, 7, 8, 10, 12]`

### Step 2a — bottleneck_experiment.py (NEW)
Sweep script to find optimal bottleneck dimension.
- For each `bottleneck_dim` in `[4, 6, 7, 8, 10, 12]`:
  1. Train autoencoder with that dim
  2. Extract bottleneck features for healthy + failed rows
  3. Train a simple Stage 2 probe classifier
  4. Record ROC-AUC, PR-AUC, F1 on held-out test set
- Print comparison table
- Save results to `DiskJson/bottleneck_sweep_results.json`

### Step 2b — train_bottleneck_classifier.py (NEW)
Main training script using the best dim from the sweep.
1. Load pre-trained autoencoder (or retrain if needed) with chosen `--bottleneck-dim`
2. Freeze encoder weights
3. Extract bottleneck features: `encoder.predict(X_healthy_scaled)` + `encoder.predict(X_failed_scaled)`
4. Build labeled dataset: healthy → 0, failed → 1
5. Handle class imbalance via `class_weight` in Keras fit
6. Train Stage 2 feedforward NN
7. Find optimal threshold on PR curve (F1-maximizing, not fixed 0.5)
8. Evaluate: ROC-AUC, PR-AUC, confusion matrix, F1 per class
9. Save `disk_bottleneck_classifier.keras` + `bottleneck_metadata.json`

### Step 2c — predict_bottleneck.py (NEW)
Single-disk inference script for Implementation 2.
- Same input interface as `predict_autoencoder.py` (path to smartctl JSON)
- **Loads:** `disk_encoder.keras` + `disk_bottleneck_classifier.keras` + `tf_scaler.pkl` + `bottleneck_metadata.json`
- **Pipeline:** raw JSON → preprocess → scale → encoder → classifier → P(failure) → threshold
- **Output:**
  ```json
  {
    "anomaly": true,
    "failure_probability": 0.87,
    "threshold": 0.42,
    "verdict": "ANOMALY_DETECTED",
    "bottleneck_features": [0.12, 0.88, 0.03, 0.55, 0.71, 0.14, 0.39]
  }
  ```

### Impl 2 — File Structure
```
srcML/tensorflow_classification/     ← NEW package
├── __init__.py
├── bottleneck_experiment.py         ← sweep bottleneck dims, find best
├── train_bottleneck_classifier.py   ← Stage 2 supervised training
└── predict_bottleneck.py            ← single-disk inference

srcML/tensorflow_classification/ artifacts:
├── disk_bottleneck_classifier.keras
└── bottleneck_metadata.json

srcML/tensorflow_anomaly/ artifacts (reused from Impl 1):
├── disk_encoder.keras               ← frozen encoder
└── tf_scaler.pkl
```

---

## Execution Order

```
[x] Step 1  →  Modify train_autoencoder.py
               - Add --bottleneck-dim arg (default 12)
               - Fix defaults: batch=128, healthy-per-file=750
               - Save disk_encoder.keras after training

[x] Step 2  →  Create predict_autoencoder.py
               - Single-disk inference for Impl 1
               - Test against DiskJson/ files

[ ] Step 3  →  Retrain autoencoder with corrected defaults
               - Verify results match run 05 quality (ROC-AUC ~0.90)
               - Impl 1 is COMPLETE

[x] Step 4  →  Create bottleneck_experiment.py  (tensorflow_classification/)
               - Run sweep: bottleneck_dim in [4, 6, 7, 8, 10, 12]
               - Pick best dim (expected: 6-8)

[x] Step 5  →  Create train_bottleneck_classifier.py  (tensorflow_classification/)
               - Stage 2 supervised training on bottleneck features
               - Tune threshold on PR curve

[x] Step 6  →  Create predict_bottleneck.py  (tensorflow_classification/)
               - Single-disk inference for Impl 2
               - Test against DiskJson/ files

[ ] Step 7  →  Compare Impl 1 vs Impl 2 results side by side
               - Expected: Impl 2 significantly improves failure recall
```

---

## Notes

- Leave `disk_pipeline.py`, `disk_health_pipeline.pkl`, and `backend/` untouched
- All new scripts use `argparse` — no hardcoded paths
- Both `predict_*.py` scripts accept the same JSON input format (smartctl -j output)
- Bottleneck dim sweep results saved to `DiskJson/` alongside other run results
