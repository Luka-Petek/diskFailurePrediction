# Disk Failure Prediction — Plan

## Autoencoder Experimental History (referenca)

| Run | Healthy Rows | Batch | Epochs | ROC-AUC | PR-AUC | Failure Recall |
|-----|-------------|-------|--------|---------|--------|----------------|
| 01  | 146K        | 1024  | 67     | 0.861   | 0.590  | 25.4%          |
| 02  | 292K        | 512   | 39     | 0.861   | 0.554  | 43.4%          |
| 03  | 438K        | 2048  | 60     | 0.783   | 0.201  | 12.8% ← worst  |
| 04  | 292K        | 256   | 58     | 0.882   | 0.568  | 43.7%          |
| 05  | 292K        | 128   | 37     | 0.901   | 0.600  | 44.7% ← best   |

**Key finding:** 292K healthy rows + batch_size=128 je optimum.

## Model Performance Summary (trenutno stanje)

| Model | ROC-AUC | Failure Recall | Failure F1 | Status |
|-------|---------|----------------|------------|--------|
| sklearn Random Forest | — | 0.86 | 0.90 | ✅ done, nespremenjen |
| TF Autoencoder (Impl 1) | 0.901 | 0.447 | 0.555 | ✅ done |
| TF Bottleneck Classifier (Impl 2) | 0.929 | 0.891 | 0.887 | ✅ done |
| K-Means na bottleneck | — | — | — | ⬜ Step 8 |
| HDBSCAN na bottleneck | — | — | — | ⬜ Step 9 |

---

# PREDZADNJI KORAK — Clustering Implementations

## Step 8 — K-Means na bottleneck features

**File:** `srcML/tensorflow_classification/cluster_bottleneck.py`

### What it does
- Naloži zamrznjeni Impl 2 encoder (`disk_clf_encoder.keras`) + scaler (`clf_scaler.pkl`)
- Vzame vzorec podatkov iz DiskData (healthy + failure)
- Ekstrahira 8-dim bottleneck features za vse vrstice
- Najprej požene Elbow method (k=2..10) + Silhouette score → shrani graf `Graphs/kmeans_elbow.png`
- Na podlagi grafa izbere optimalni k (pričakovano 3–5), potem fit K-Means z izbranim k → shrani `clf_kmeans.pkl`
- Po fittingu označi vsak cluster z risk labelom glede na delež failure vrstic v njem:
  - cluster z >50% failure → `HIGH_RISK` (risk_score=1.0)
  - cluster z 10-50% failure → `ELEVATED_RISK` (risk_score=0.5)
  - cluster z <10% failure → `LOW_RISK` (risk_score=0.0)
- Shrani cluster_labels mapping v `clf_kmeans_metadata.json`
- Generira 2D PCA scatter plot → `Graphs/bottleneck_kmeans_clusters.png`

### Artifacts
```
srcML/tensorflow_classification/
├── clf_kmeans.pkl                ← fitted KMeans(k=3) na bottleneck features
└── clf_kmeans_metadata.json      ← cluster → risk_label + failure_rate mapping
Graphs/
└── bottleneck_kmeans_clusters.png
```

### Inference (za Combined API)
```python
# cluster_score ∈ {0.0, 0.5, 1.0} glede na cluster membership
bottleneck = encoder.predict(X_scaled)
cluster_id = kmeans.predict(bottleneck)[0]
cluster_score = metadata["cluster_risk"][str(cluster_id)]["risk_score"]
```

---

## Step 9 — UMAP + HDBSCAN

**File:** `srcML/tensorflow_classification/umap_hdbscan.py`

### What it does
- Isti bottleneck features kot Step 8
- UMAP: 8D → 2D (n_neighbors=30, min_dist=0.1, metric='euclidean')
- HDBSCAN: avtomatično odkriva število clusterjev (min_cluster_size=50)
  - cluster=-1 pomeni outlier (izjemno visok risk signal!)
- Scatter plot: točke barvane po HDBSCAN clusteru + failure label kot marker shape
- Shrani: `Graphs/umap_hdbscan.png`
- Analizira vsak cluster: delež failure vrstic → risk mapping (enako kot K-Means)
- Shrani `hdbscan_metadata.json` (cluster → failure_rate, outlier_failure_rate)

### Key insight za Combined API
Disk ki ga HDBSCAN označi kot **outlier (cluster=-1)** → posebej visok risk bonus
ker outlier v latentnem prostoru = "ni podoben nobeni normalnim skupini"

### Artifacts
```
srcML/tensorflow_classification/
└── hdbscan_metadata.json         ← cluster → failure_rate mapping
Graphs/
└── umap_hdbscan.png
```

### Packages needed
```
pip install umap-learn hdbscan
```

---

---

# ZADNJI KORAK — Multi-Model API Exposure

## Arhitektura: 5 endpoints

```
POST /api/predict/anomaly          ← Impl 1: TF Autoencoder
POST /api/predict/classification   ← Impl 2: TF Bottleneck Classifier
POST /api/predict/clustering       ← K-Means na bottleneck features
POST /api/predict/sklearn          ← sklearn Random Forest
POST /api/predict/combined         ← VSI 4 + kombinirana formula ← GLAVNA
```

Vsi sprejmejo isti input: `smartctl -j` JSON file (multipart upload).

---

## Step 10 — Individualni endpoints (za testiranje)

### `/api/predict/anomaly`
Kliče logiko iz `predict_autoencoder.py`.
```json
{
  "anomaly": true,
  "anomaly_score": 0.73,
  "reconstruction_error": 0.031,
  "threshold": 0.00407,
  "verdict": "ANOMALY_DETECTED"
}
```

### `/api/predict/classification`
Kliče logiko iz `predict_bottleneck.py`.
```json
{
  "failure_predicted": true,
  "failure_probability": 0.87,
  "threshold": 0.2825,
  "verdict": "FAILURE",
  "bottleneck_features": [5.44, 0.0, 0.0, 0.36, 0.0, 4.67, 2.83, 0.0]
}
```

### `/api/predict/clustering`
Kliče K-Means inference na bottleneck features.
```json
{
  "cluster_id": 2,
  "cluster_label": "HIGH_RISK",
  "cluster_failure_rate": 0.71,
  "cluster_score": 1.0
}
```

### `/api/predict/sklearn`
Kliče obstoječi `DiskHealthPipeline.analyze()` iz `disk_pipeline.py`.
```json
{
  "hir_risk_score": 82.4,
  "verdict": "Critical",
  "models_output": {
    "classification_fail": true,
    "predicted_smart_5_sectors": 312.0,
    "cluster_profile_id": 1
  }
}
```

---

## Step 11 — Combined endpoint `/api/predict/combined`

### Logika
Požene VSE 4 modele zaporedno na istem inputu, potem izračuna skupni `DISK_HEALTH_SCORE`.

### Utežena formula
```
S_clf   = failure_probability          (Impl 2, float 0..1)
S_anom  = anomaly_score                (Impl 1, float 0..1, normalized)
S_clust = cluster_score                (K-Means risk, {0.0, 0.5, 1.0})
S_skl   = sklearn_fail_probability     (RF predict_proba[:,1], float 0..1)

W_clf   = 0.50   ← največ: best performance (F1=0.887, ROC-AUC=0.929)
W_anom  = 0.10   ← najmanj: low recall (44.7%), samo confirmation signal
W_clust = 0.20   ← TBD: nastavi se po evalvaciji clustering performance
W_skl   = 0.20   ← dobri rezultati (sklearn RF F1~0.90), zanesljiv sekundarni

DISK_HEALTH_SCORE = W_clf*S_clf + W_anom*S_anom + W_clust*S_clust + W_skl*S_skl
```

### Utežen verdict
| DISK_HEALTH_SCORE | Verdict |
|---|---|
| ≥ 0.70 | `FAILURE` |
| 0.40 – 0.70 | `AT_RISK` |
| < 0.40 | `HEALTHY` |

### Output
```json
{
  "disk_health_score": 0.76,
  "verdict": "FAILURE",
  "confidence": "high",
  "model_scores": {
    "tf_classification": { "failure_probability": 0.87, "verdict": "FAILURE", "weight": 0.50 },
    "tf_anomaly":        { "anomaly_score": 0.73,       "verdict": "ANOMALY_DETECTED", "weight": 0.10 },
    "clustering":        { "cluster_score": 1.0,        "cluster_label": "HIGH_RISK", "weight": 0.20 },
    "sklearn":           { "failure_probability": 0.81, "verdict": "Critical", "weight": 0.20 }
  },
  "consensus": {
    "models_agreeing_failure": 4,
    "models_total": 4
  }
}
```

### `confidence` logika
- `"high"` — vsi 4 modeli se strinjajo
- `"medium"` — 3/4 se strinjajo
- `"low"` — 2/4 ali manj se strinjajo (dodaj opozorilo v frontend)

---

## Step 12 — Backend refactoring

`backend/main.py` trenutno vsebuje samo en endpoint za sklearn pipeline.
Treba razširiti na 5 endpointov + naložiti vse modele ob zagonu.

### Model loading ob startu (enkrat, ne per-request)
```python
# Impl 1
autoencoder = tf.keras.models.load_model(...)
ae_scaler = joblib.load(...)
ae_metadata = json.load(...)

# Impl 2
clf_encoder = tf.keras.models.load_model(...)
clf_classifier = tf.keras.models.load_model(...)
clf_scaler = joblib.load(...)
clf_metadata = json.load(...)

# Clustering
kmeans = joblib.load(...)
kmeans_metadata = json.load(...)

# sklearn
sklearn_pipeline = joblib.load(...)  # obstoječi DiskHealthPipeline
```

### Dependency injection pattern
Vsi modeli se naložijo enkrat in posredujejo endpointom kot app.state — ne globalne spremenljivke.

---

## Execution Checklist (implementacija v prihodnje)

```
[ ] Step 8  → cluster_bottleneck.py  (K-Means na bottleneck)
              - pip install: /
              - Input: DiskData CSV + Impl 2 encoder
              - Output: clf_kmeans.pkl, clf_kmeans_metadata.json, Graphs/bottleneck_kmeans_clusters.png

[ ] Step 9  → umap_hdbscan.py  (UMAP + HDBSCAN vizualizacija)
              - pip install: umap-learn hdbscan
              - Input: DiskData CSV + Impl 2 encoder
              - Output: hdbscan_metadata.json, Graphs/umap_hdbscan.png

[ ] Step 10 → backend/main.py — 4 individualni endpoints
              - Refaktoriraj obstoječi /api/analyze-smart-json → /api/predict/sklearn
              - Dodaj /api/predict/anomaly, /api/predict/classification, /api/predict/clustering
              - Model loading: app.state pattern

[ ] Step 11 → /api/predict/combined endpoint
              - Požene vse 4 modele
              - Izračuna DISK_HEALTH_SCORE z uteženo formulo
              - Vrne model_scores + consensus + confidence

[ ] Step 12 → Testiranje vseh 5 endpointov z DiskJson/ primeri
              - Pričakovano: classification in sklearn se strinjata v večini primerov
              - Edge case: anomaly detection odkrije, classification ne → "low confidence"
```
