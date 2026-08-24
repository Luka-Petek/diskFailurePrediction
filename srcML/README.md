# srcML — ML Pipeline Engineering Notes

Detailed description of the ML process across all 4 implementations, the shared preprocessing, and the AHI fusion. Written for engineering review — every claim here is traceable to code in this directory.

---

## Pipeline Overview

```
Backblaze daily CSVs (32M+ rows, 365 files)
         │
         ▼
┌─────────────────────────┐
│   Shared Preprocessing   │
│   procesiraj_podatke()   │
│   → 19 FEATURE_COLUMNS   │
└──────────┬──────────────┘
           │
    ┌──────┼──────┬──────────┐
    ▼      ▼      ▼          ▼
  Impl 0  Impl 1  Impl 2    Impl C
  RF      AE      AE→Clf    UMAP+HDBSCAN
    │      │      │          │
    ▼      ▼      ▼          ▼
  P(fail) score  P(fail)   cluster risk
    │      │      │          │
    └──────┴──┬───┴──────────┘
              ▼
         AHI RMS fusion
              │
              ▼
        Score [3–97]
```

---

## Shared Preprocessing

### `srcML/sklearn/disk_pipeline.py` — `procesiraj_podatke(df_raw)`

Called by `prepare_features()` before every model. Single source of truth for feature extraction.

1. **Capacity**: `capacity_bytes` → `capacity_gigabytes` (divide by 1024³)
2. **SSD detection**: keyword matching on model name (`'SSD', 'MTFD', 'SSDSC', '850 PRO', '870 EVO', '860 PRO', '5300'`). If match → `jeSSD = 1`, else `0`.
3. **NaN filling** (SSD-aware):
   - Error counters (`smart_1`, `smart_7`, `smart_191`, `smart_192`): fill with 0
   - Mechanical attributes (`smart_3`, `smart_4`, `smart_193`): if SSD → 0, if HDD → hardcoded medians (0, 15, 1200)
   - `smart_12` (power cycles): fill with median 12
   - Critical attrs (`smart_5`, `smart_9`, `smart_187`, `smart_188`, `smart_197`, `smart_198`): fill with 0
4. **Feature engineering**:
   - `any_critical_error` = 1 if sum of (smart_5, smart_187, smart_197, smart_198) > 0
   - `total_error_count` = sum of those 4 attributes
   - `error_per_gb` = `total_error_count / (capacity_gigabytes + 1e-5)`
5. **Dropped**: `smart_190`, `smart_194`, `smart_199`, `smart_10` (non-informative)

### `srcML/nn_preprocessing/preprocessing.py` — `prepare_features(df_raw)`

Calls `procesiraj_podatke`, then extracts 19 `FEATURE_COLUMNS`. Missing columns → 0.0. Replaces inf → NaN → 0.0. Returns float32 DataFrame.

### Dataset builders

- **`build_dataset_from_many_csvs`**: samples N healthy + M failure rows per CSV file. Used by Impl 1 AE (healthy-only training).
- **`build_balanced_dataset_from_csvs`**: two-pass. Pass 1 collects ALL failure rows (from dedicated `csv/vseOdpovedi.csv` or by scanning). Pass 2 collects equal number of healthy rows. Result: 50:50 balanced. Used by Impl 2 classifier and Impl C clustering.

---

## Implementation 0 — Random Forest

**File**: `srcML/sklearn/smart_scan_model.ipynb` (training) · `disk_pipeline.py` (inference)

- **Model**: sklearn RandomForestClassifier on 19 SMART features + manufacturer one-hot encoding
- **Training**: done in notebook; pipeline serialized as `disk_health_pipeline.pkl`
- **Inference**: `DiskHealthPipeline.analyze()` — parses JSON, preprocesses, one-hot encodes manufacturer (Seagate, WD, HGST, Toshiba, Samsung, Crucial, Other), predicts `predict_proba`, clamps to [5, 97]
- **Output**: `failure_probability` ∈ [0, 1], `hir_risk_score` ∈ [5, 97], verdict

---

## Implementation 1 — Autoencoder (Anomaly Detection)

**File**: `srcML/tensorflow_anomaly/train_autoencoder.py`

### Architecture
```
Input(19) → Dense(64, relu) → BatchNorm → Dropout(0.10)
         → Dense(32, relu) → BatchNorm
         → Dense(12, relu)  ← bottleneck
         → Dense(32, relu) → BatchNorm
         → Dense(64, relu)
         → Dense(19, sigmoid)  ← reconstruction
```

### Training
- **Data**: healthy rows only, sampled via `build_dataset_from_many_csvs` (1000 healthy/file, 100 failure/file for eval)
- **Split**: 80/20 train/val on healthy rows (`train_test_split`, seed 42)
- **Scaler**: `MinMaxScaler` fit on train
- **Loss**: MAE (mean absolute error between input and reconstruction)
- **Optimizer**: Adam, lr=0.001
- **Callbacks**: EarlyStopping(`val_loss`, patience=10, restore_best_weights), ReduceLROnPlateau(`val_loss`, factor=0.5, patience=4, min_lr=1e-6), TensorBoard
- **Batch size**: 128, Epochs: 60

### Threshold & scoring
- **Threshold**: 99th percentile of validation reconstruction errors
- **Normalization**: `score = (error - threshold) / (p999 - threshold)`, clipped to [0, 1]
- **Anomaly flag**: `error > threshold`

### Evaluation
- ROC-AUC and PR-AUC computed on: validation healthy errors (label=0) + failure errors (label=1)
- Results: ROC-AUC 0.901, PR-AUC 0.600, failure recall 44.7% at 1% FPR

### Artifacts exported
- `disk_autoencoder.keras` — full AE model
- `disk_encoder.keras` — encoder submodel (input → bottleneck)
- `tf_scaler.pkl` — fitted MinMaxScaler
- `tf_metadata.json` — threshold, percentiles (p95/p99/p999), evaluation metrics, training config

---

## Implementation 2 — Bottleneck Classifier (2-stage)

### Stage 1: Feature Extractor AE

**File**: `srcML/tensorflow_classification/train_autoencoder.py`

Same architecture as Impl 1 AE but with **bottleneck_dim = 8** (determined empirically). Trained on healthy rows only, same MinMaxScaler + MAE loss. Exports:
- `disk_clf_encoder.keras` — frozen encoder (19 → 8 dim)
- `clf_scaler.pkl` — fitted MinMaxScaler
- `clf_ae_metadata.json` — bottleneck_dim, training config

### Stage 2: Supervised Classifier

**File**: `srcML/tensorflow_classification/train_bottleneck_classifier.py`

#### Architecture
```
Input(8) → Dense(16, relu) → Dropout(0.2) → Dense(8, relu) → Dense(1, sigmoid)
```

#### Training
- **Encoder**: loaded from Stage 1, frozen (`encoder.trainable = False`)
- **Data**: 50:50 balanced via `build_balanced_dataset_from_csvs` (4,414 failures : 4,414 healthy)
- **Feature extraction**: raw → `prepare_features` → scaler.transform → encoder.predict → 8-dim bottleneck
- **Split**: 70/15/15 stratified (`train_test_split`, seed 42)
- **Class weights**: `compute_class_weight("balanced")` — compensates if split isn't perfectly 50:50
- **Loss**: binary_crossentropy
- **Optimizer**: Adam, lr=0.001
- **Metrics**: accuracy, AUC
- **Callbacks**: EarlyStopping(`val_auc`, patience=15, mode=max, restore_best_weights), ReduceLROnPlateau(`val_auc`, factor=0.5, patience=5, min_lr=1e-6), TensorBoard
- **Batch size**: 128, Epochs: 100

#### Threshold tuning
- **Threshold**: F1-optimal on validation set via `precision_recall_curve` — finds threshold that maximizes F1
- **High-risk threshold**: hardcoded 0.65 (separate "FAILURE" verdict from "AT_RISK")

#### Evaluation (on test set, unseen during training/threshold tuning)
- ROC-AUC 0.9289, PR-AUC 0.9337, failure recall 89.1%, failure F1 88.7%

#### Artifacts exported
- `disk_bottleneck_classifier.keras` — classifier model
- `bottleneck_metadata.json` — threshold, high_risk_threshold, evaluation metrics, training config, feature_columns

---

## Implementation C — UMAP + HDBSCAN Clustering

**File**: `srcML/tensorflow_clustering/umap_hdbscan.py`

### Process
1. Load Impl 2 encoder + scaler (shared artifacts)
2. Build 50:50 balanced dataset (same `build_balanced_dataset_from_csvs`)
3. Extract 8-dim bottleneck features via encoder
4. **UMAP**: reduce 8D → 2D for visualization only (n_neighbors=30, min_dist=0.1, euclidean)
5. **HDBSCAN**: cluster on full 8D bottleneck (not UMAP 2D). min_cluster_size=50, euclidean, `prediction_data=True` for `approximate_predict` on new points

### Cluster risk scoring (`analyze_clusters`)
For each cluster:
- `failure_rate = failures / total` (computed on the balanced 50:50 dataset)
- `risk_score = failure_rate` (raw value)
- `risk_label`: HIGH_RISK (>50%), ELEVATED_RISK (10–50%), LOW_RISK (<10%), OUTLIER (cluster_id=-1)

### Artifacts exported
- `clf_hdbscan.pkl` — fitted HDBSCAN clusterer (with `prediction_data` for inference)
- `clf_umap_reducer.pkl` — UMAP reducer (for re-visualization, not needed for inference)
- `hdbscan_metadata.json` — n_clusters, outlier_ratio, per-cluster risk metadata
- TensorBoard Embedding Projector logs (bottleneck embeddings colored by cluster + failure label)

### Inference
`hdbscan.approximate_predict(clusterer, bottleneck)` → cluster_id → look up `risk_score` from metadata.

---

## AHI Fusion - Formula

![AHI Formula](../Graphs/hir_formula.png)

Since weights sum to 1.0, the denominator (Σw) is omitted.

### Weights and inputs

| Symbol | Source | Weight | Output range |
|---|---|---|---|
| K | RF `predict_proba` | 0.30 | [0, 1] probability |
| R | Bottleneck classifier sigmoid | 0.40 | [0, 1] probability |
| A | AE normalized anomaly score | 0.20 | [0, 1] (error-threshold)/(p999-threshold) |
| C | HDBSCAN cluster risk_score | 0.10 | [0, 1] cluster failure rate |

### Verdict thresholds
- HEALTHY: < 40
- WARNING: 40–75
- CRITICAL: > 75

### Why RMS over linear average
RMS amplifies large individual signals. A disk scoring 0.9 on one model and 0.1 on others gets `sqrt(0.4·0.9²) ≈ 0.57` with RMS vs `0.4·0.9 = 0.36` with linear. A catastrophic signal on one axis cannot be averaged away.

### Inference flow (`hir_final.py`)
1. Load all 4 model artifacts (RF pipeline, encoder+classifier+scaler, AE+scaler, HDBSCAN+metadata)
2. Parse smartctl JSON → `pretvori_json_v_surovi_df` → raw DataFrame
3. Score each model independently (all share the same `prepare_features` preprocessing)
4. RMS fusion → clamp → verdict

---

## Backend inference

**File**: `backend/main.py`

FastAPI app with `lifespan` context that loads all artifacts on startup. Same AHI computation as `hir_final.py`.

### Endpoints
- `POST /api/predict/anomaly` — Impl 1 AE only
- `POST /api/predict/classification` — Impl 2 classifier only
- `POST /api/predict/clustering` — Impl C HDBSCAN only (runs classifier first for bottleneck)
- `POST /api/predict/sklearn` — Impl 0 RF only
- `POST /api/predict/combined` — all 4 models, AHI fusion, returns per-model breakdown + consensus

### Graceful degradation
Each model loads in its own try/except. If an artifact is missing, that model is set to `None` and skipped during combined inference. The AHI formula renormalizes weights over active models only (`active_weight` tracks sum of weights of models that succeeded).

---

## Reproducibility

- Seeds: `np.random.seed`, `random.seed`, `tf.random.set_seed` — all set to 42 in every training script
- Every training script exports a metadata JSON with: feature_columns, training config (rows, epochs, batch_size, class weights), evaluation metrics, thresholds
- Scaler (`.pkl`), model (`.keras`), metadata (`.json`) saved together per model
- sklearn pinned for pickle compatibility (pipeline was saved with specific sklearn version)
