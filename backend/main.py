import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import hdbscan as hdbscan_lib
import joblib
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

SRCML_PATH = "/app/srcML"
if SRCML_PATH not in sys.path:
    sys.path.insert(0, SRCML_PATH)

#pkl datoteke so bile shranjene ko je bil disk_pipeline.py v rootu — dodamo srcML/sklearn v path
SKLEARN_PATH = "/app/srcML/sklearn"
if SKLEARN_PATH not in sys.path:
    sys.path.insert(0, SKLEARN_PATH)

APP_ROOT = Path("/app")
AE_DIR = APP_ROOT / "srcML" / "tensorflow_anomaly"
CLF_DIR = APP_ROOT / "srcML" / "tensorflow_classification"
CLUSTER_DIR = APP_ROOT / "srcML" / "tensorflow_clustering"
SKLEARN_PIPELINE_PATH = APP_ROOT / "srcML" / "sklearn" / "disk_health_pipeline.pkl"

from srcML.sklearn.disk_pipeline import pretvori_json_v_surovi_df
from srcML.nn_preprocessing.preprocessing import prepare_features


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.ae_model = None
    app.state.ae_scaler = None
    app.state.ae_metadata = None
    app.state.clf_encoder = None
    app.state.clf_classifier = None
    app.state.clf_scaler = None
    app.state.clf_metadata = None
    app.state.hdbscan = None
    app.state.hdbscan_metadata = None
    app.state.sklearn_pipeline = None

    #--- Impl 1: TF Autoencoder ---
    try:
        app.state.ae_model = tf.keras.models.load_model(AE_DIR / "disk_autoencoder.keras")
        app.state.ae_scaler = joblib.load(AE_DIR / "tf_scaler.pkl")
        with open(AE_DIR / "tf_metadata.json", encoding="utf-8") as f:
            app.state.ae_metadata = json.load(f)
        print("Impl 1 (Autoencoder) naložen.")
    except Exception as e:
        app.state.ae_model = None
        print(f"Impl 1 ni na voljo: {e}")

    #--- Impl 2: TF Bottleneck Classifier ---
    try:
        app.state.clf_encoder = tf.keras.models.load_model(CLF_DIR / "disk_clf_encoder.keras")
        app.state.clf_classifier = tf.keras.models.load_model(CLF_DIR / "disk_bottleneck_classifier.keras")
        app.state.clf_scaler = joblib.load(CLF_DIR / "clf_scaler.pkl")
        with open(CLF_DIR / "bottleneck_metadata.json", encoding="utf-8") as f:
            app.state.clf_metadata = json.load(f)
        print("Impl 2 (Bottleneck Classifier) naložen.")
    except Exception as e:
        app.state.clf_encoder = None
        print(f"Impl 2 ni na voljo: {e}")

    #--- HDBSCAN clustering na bottleneck ---
    try:
        app.state.hdbscan = joblib.load(CLUSTER_DIR / "clf_hdbscan.pkl")
        with open(CLUSTER_DIR / "hdbscan_metadata.json", encoding="utf-8") as f:
            app.state.hdbscan_metadata = json.load(f)
        print("HDBSCAN model naložen.")
    except Exception as e:
        app.state.hdbscan = None
        print(f"HDBSCAN ni na voljo (poženite umap_hdbscan.py): {e}")

    #--- sklearn Pipeline ---
    try:
        app.state.sklearn_pipeline = joblib.load(SKLEARN_PIPELINE_PATH)
        print("sklearn Pipeline naložen.")
    except Exception as e:
        app.state.sklearn_pipeline = None
        print(f"sklearn Pipeline ni na voljo: {e}")

    yield


app = FastAPI(title="DiskGuard API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _parse_upload(raw_bytes: bytes) -> dict:
    try:
        return json.loads(raw_bytes)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Neveljavna JSON datoteka: {e}")


def _require(model, name: str):
    if model is None:
        raise HTTPException(status_code=503, detail=f"Model ni na voljo: {name}")


def _infer_anomaly(app_state, smartctl_dict: dict) -> dict:
    _require(app_state.ae_model, "TF Autoencoder")
    raw_df = pretvori_json_v_surovi_df(smartctl_dict)
    X = prepare_features(raw_df)
    X_scaled = app_state.ae_scaler.transform(X).astype("float32")
    reconstructed = app_state.ae_model.predict(X_scaled, batch_size=1, verbose=0)
    reconstruction_error = float(np.mean(np.abs(X_scaled - reconstructed)))

    threshold = app_state.ae_metadata["threshold"]
    p999 = app_state.ae_metadata["normalization"]["score_p999"]
    anomaly = reconstruction_error > threshold

    if p999 <= threshold:
        anomaly_score = 0.0
    else:
        anomaly_score = float(np.clip((reconstruction_error - threshold) / (p999 - threshold), 0.0, 1.0))

    return {
        "anomaly": bool(anomaly),
        "anomaly_score": round(anomaly_score, 4),
        "reconstruction_error": round(reconstruction_error, 6),
        "threshold": round(threshold, 6),
        "verdict": "ANOMALY_DETECTED" if anomaly else "HEALTHY",
    }


def _infer_classification(app_state, smartctl_dict: dict) -> tuple[dict, np.ndarray]:
    _require(app_state.clf_encoder, "TF Bottleneck Classifier")
    raw_df = pretvori_json_v_surovi_df(smartctl_dict)
    X = prepare_features(raw_df)
    X_scaled = app_state.clf_scaler.transform(X).astype("float32")
    bottleneck = app_state.clf_encoder.predict(X_scaled, batch_size=1, verbose=0)
    failure_prob = float(app_state.clf_classifier.predict(bottleneck, batch_size=1, verbose=0).flatten()[0])

    threshold = app_state.clf_metadata["threshold"]
    HIGH_RISK_THRESHOLD = 0.65

    if failure_prob >= HIGH_RISK_THRESHOLD:
        verdict = "FAILURE"
        failure_predicted = True
    elif failure_prob >= threshold:
        verdict = "AT_RISK"
        failure_predicted = True
    else:
        verdict = "HEALTHY"
        failure_predicted = False

    result = {
        "failure_predicted": failure_predicted,
        "failure_probability": round(failure_prob, 4),
        "threshold": round(threshold, 4),
        "high_risk_threshold": HIGH_RISK_THRESHOLD,
        "verdict": verdict,
        "bottleneck_features": [round(float(v), 4) for v in bottleneck.flatten()],
    }
    return result, bottleneck


def _infer_clustering(app_state, bottleneck: np.ndarray) -> dict:
    _require(app_state.hdbscan, "HDBSCAN clustering")
    labels, strengths = hdbscan_lib.approximate_predict(app_state.hdbscan, bottleneck)
    cluster_id = int(labels[0])
    strength = float(strengths[0])
    risk_info = app_state.hdbscan_metadata["cluster_risk"].get(str(cluster_id), {})
    return {
        "cluster_id": cluster_id,
        "is_outlier": cluster_id == -1,
        "cluster_strength": round(strength, 4),
        "cluster_label": risk_info.get("risk_label", "UNKNOWN"),
        "cluster_failure_rate": risk_info.get("failure_rate", 0.0),
        "cluster_score": risk_info.get("risk_score", 0.0),
    }


def _infer_sklearn(app_state, smartctl_dict: dict) -> dict:
    _require(app_state.sklearn_pipeline, "sklearn Pipeline")
    raw_df = pretvori_json_v_surovi_df(smartctl_dict)
    return app_state.sklearn_pipeline.analyze(raw_df)


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.post("/api/predict/anomaly")
async def predict_anomaly(request: Request, file: UploadFile = File(...)):
    raw = await file.read()
    smartctl_dict = _parse_upload(raw)
    return _infer_anomaly(request.app.state, smartctl_dict)


@app.post("/api/predict/classification")
async def predict_classification(request: Request, file: UploadFile = File(...)):
    raw = await file.read()
    smartctl_dict = _parse_upload(raw)
    result, _ = _infer_classification(request.app.state, smartctl_dict)
    return result


@app.post("/api/predict/clustering")
async def predict_clustering(request: Request, file: UploadFile = File(...)):
    raw = await file.read()
    smartctl_dict = _parse_upload(raw)
    _, bottleneck = _infer_classification(request.app.state, smartctl_dict)
    return _infer_clustering(request.app.state, bottleneck)


@app.post("/api/predict/sklearn")
async def predict_sklearn(request: Request, file: UploadFile = File(...)):
    raw = await file.read()
    smartctl_dict = _parse_upload(raw)
    return _infer_sklearn(request.app.state, smartctl_dict)


@app.post("/api/predict/combined")
async def predict_combined(request: Request, file: UploadFile = File(...)):
    raw = await file.read()
    smartctl_dict = _parse_upload(raw)
    state = request.app.state

    W_clf, W_anom, W_clust, W_skl = 0.40, 0.20, 0.10, 0.30
    model_scores = {}
    weighted_sum_sq = 0.0
    active_weight = 0.0
    models_predicting_failure = 0
    models_total = 0

    #--- Impl 2: Bottleneck Classifier ---
    try:
        clf_result, bottleneck = _infer_classification(state, smartctl_dict)
        s_clf = clf_result["failure_probability"]
        weighted_sum_sq += W_clf * s_clf ** 2
        active_weight += W_clf
        models_total += 1
        if clf_result["failure_predicted"]:
            models_predicting_failure += 1
        model_scores["tf_classification"] = {
            "failure_probability": clf_result["failure_probability"],
            "verdict": clf_result["verdict"],
            "weight": W_clf,
            "threshold": clf_result["threshold"],
            "high_risk_threshold": clf_result["high_risk_threshold"],
            "bottleneck_features": clf_result["bottleneck_features"],
        }
    except HTTPException:
        clf_result = None
        bottleneck = None

    #--- Impl 1: Autoencoder ---
    try:
        ae_result = _infer_anomaly(state, smartctl_dict)
        s_anom = ae_result["anomaly_score"]
        weighted_sum_sq += W_anom * s_anom ** 2
        active_weight += W_anom
        models_total += 1
        if ae_result["anomaly"]:
            models_predicting_failure += 1
        model_scores["tf_anomaly"] = {
            "anomaly_score": ae_result["anomaly_score"],
            "verdict": ae_result["verdict"],
            "weight": W_anom,
            "reconstruction_error": ae_result["reconstruction_error"],
            "threshold": ae_result["threshold"],
            "is_anomaly": ae_result["anomaly"],
        }
    except HTTPException:
        pass

    #--- HDBSCAN Clustering ---
    if bottleneck is not None and state.hdbscan is not None:
        try:
            clust_result = _infer_clustering(state, bottleneck)
            s_clust = clust_result["cluster_score"]
            weighted_sum_sq += W_clust * s_clust ** 2
            active_weight += W_clust
            models_total += 1
            if s_clust >= 0.5:
                models_predicting_failure += 1
            model_scores["clustering"] = {
                "cluster_score": clust_result["cluster_score"],
                "cluster_label": clust_result["cluster_label"],
                "weight": W_clust,
                "cluster_id": clust_result["cluster_id"],
                "is_outlier": clust_result["is_outlier"],
                "cluster_strength": clust_result["cluster_strength"],
                "cluster_failure_rate": clust_result["cluster_failure_rate"],
            }
        except HTTPException:
            pass

    #--- sklearn Pipeline ---
    try:
        skl_result = _infer_sklearn(state, smartctl_dict)
        s_skl = skl_result.get("failure_probability", 0.0)
        weighted_sum_sq += W_skl * s_skl ** 2
        active_weight += W_skl
        models_total += 1
        if skl_result.get("models_output", {}).get("classification_fail", False):
            models_predicting_failure += 1
        model_scores["sklearn"] = {
            "failure_probability": s_skl,
            "hir_risk_score": skl_result.get("hir_risk_score"),
            "verdict": skl_result.get("verdict"),
            "weight": W_skl,
            "classification_fail": skl_result.get("models_output", {}).get("classification_fail", False),
        }
    except HTTPException:
        pass

    if active_weight == 0.0:
        raise HTTPException(status_code=503, detail="Nobeden model ni na voljo.")

    #RMS formula: sqrt( Σ(wi · si²) / Σw )
    rms_score = float(np.sqrt(weighted_sum_sq / active_weight))
    disk_health_score = round(float(np.clip(rms_score * 100, 3.0, 97.0)), 2)

    if disk_health_score >= 75.0:
        combined_verdict = "CRITICAL"
    elif disk_health_score >= 40.0:
        combined_verdict = "WARNING"
    else:
        combined_verdict = "HEALTHY"

    if models_total == 0:
        confidence = "none"
    elif models_predicting_failure == models_total or (models_total - models_predicting_failure) == models_total:
        confidence = "high"
    elif models_predicting_failure >= (models_total - 1):
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "disk_health_score": disk_health_score,
        "verdict": combined_verdict,
        "confidence": confidence,
        "model_scores": model_scores,
        "consensus": {
            "models_predicting_failure": models_predicting_failure,
            "models_total": models_total,
        },
    }


#Backward compatibility alias
@app.post("/api/analyze-smart-json")
async def analyze_smart_json_legacy(request: Request, file: UploadFile = File(...)):
    raw = await file.read()
    smartctl_dict = _parse_upload(raw)
    return _infer_sklearn(request.app.state, smartctl_dict)