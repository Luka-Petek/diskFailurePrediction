# Machine Learning - Hard Drive Failure Prediction Model & AI Assistant

A machine learning project that predicts hard drive failures using historical **SMART** (Self-Monitoring, Analysis, and Reporting Technology) sensor data. This system integrates Random Forest classification, regression, unsupervised clustering, a reusable real-time prediction pipeline, and a local AI assistant for result interpretation.

By learning disk degradation patterns, the model identifies risks before critical hardware failure and data loss occur.

### Key Features:
* **Dataset:** Backblaze open-source data (Year 2025).
* **Scale:** Processed 32M+ records, filtered into a balanced dataset of **8,828 instances**.
* **Methodology:** Supervised learning (Random Forest - regression and classification) and Unsupervised learning (K-Means).
* **Real-Time Prediction Pipeline:** A reusable pipeline that accepts live SMART JSON data and returns a disk health risk assessment through an API.
* **AI Integration:** Local LLM (Llama 3 via Ollama) providing natural language explanations for SMART parameters.

---

* **Model Source:** [smart_scan_model.ipynb](srcML/smart_scan_model.ipynb) — *This is the core notebook where the model is trained, evaluated, and exported for real-time prediction.*
* **Reusable Prediction Pipeline:** [disk_pipeline.py](srcML/disk_pipeline.py) — *This contains the reusable preprocessing and prediction logic used by the API.*
* **Serialized Health Pipeline:** [disk_health_pipeline.pkl](srcML/disk_health_pipeline.pkl) — *This is the exported machine learning pipeline used for real-time inference.*
* **Balanced Data Selection:** [pridobivanje_podatkovne_mnozice.ipynb](srcML/pridobivanje_podatkovne_mnozice.ipynb) - *This selects all problematic disks from the whole year 2025 (only 4414), completing the dataset with another 4414 randomly selected healthy disks. This selection is functional, but not optimal yet; a more efficient sampling strategy is planned.*
* **SMART scan JSON input:** [disk_data_sda.json](disk_data_sda.json) - *Example SMART scan exported from smartctl in JSON format and used for testing real-time API prediction.*

---

## Current Project Structure

The project is currently divided into multiple parts:

### Main folders:
* **backend:** FastAPI application that exposes the machine learning model through an API.
* **frontend:** React/Vite dashboard application. It is meant to display disk analytics and will use the backend API for real-time prediction.
* **srcML:** Machine learning notebooks, trained models, reusable prediction pipeline, and exported `.pkl` files.
* **csv:** Prepared datasets and intermediate data files.
* **Graphs:** Model evaluation graphs and visual explanation assets used in this README.

---

## Real-Time Prediction API

The project now includes a **FastAPI backend** that allows real-time disk health prediction from SMART JSON data.

The API endpoint accepts a SMART JSON file, converts it into the model-compatible structure, runs preprocessing, applies the trained pipeline, and returns a health prediction result.

### Example request (currently via cli, latter will be displayed on dashboard):

bash curl -X POST "[http://localhost:8000/api/analyze-smart-json](http://localhost:8000/api/analyze-smart-json)" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@disk_data_sda.json;type=application/json"

### Example response:

json { "hir_risk_score": 6.76, "verdict": "Healthy", "models_output": { "classification_fail": false, "predicted_smart_5_sectors": 9.6, "cluster_profile_id": 0 } }

### Returned values:
* **hir_risk_score:** Final disk risk score calculated from classification, regression, clustering, and critical SMART error signals.
* **verdict:** Final health category: `Healthy`, `Warning`, or `Critical`.
* **classification_fail:** Binary Random Forest classification result.
* **predicted_smart_5_sectors:** Regression prediction for SMART 5 / Reallocated Sectors Count.
* **cluster_profile_id:** K-Means cluster profile assigned to the disk.

This API is intended to be used by the dashboard application in the `frontend` folder, where real-time disk scans can be uploaded and displayed in a more user-friendly visual form.

---

### Docker services:
1. **disk-ml-backend:** FastAPI backend used for machine learning inference.
2. **disk-ml-frontend:** React/Vite frontend dashboard.
3. **Ollama / LLM service:** Local AI assistant setup, used for natural language interpretation of SMART results.

The backend loads the serialized pipeline from: 

text /app/srcML/disk_health_pipeline.pkl


The `srcML` folder is mounted into the backend container, so the latest model pipeline and preprocessing code are available to the API.

---

## Model Performance & Analysis

The model demonstrates high reliability in identifying failure-prone drives using dual-method analysis:

### 1. Classification Analysis (Yes/No)
Categorizes drives into binary states: **Healthy** or **Failure-Prone**.
* **Accuracy:** 90.15%
* **Recall:** 86.00% (Critical for capturing actual failure events)
* **F1-Score:** 0.88

![classification.png](Graphs/classification.png)

### 2. Regression Analysis (SMART 5)
Predicting the value of **SMART 5 (Reallocated Sectors Count)**.
* **Predictive Forecasting:** Instead of a binary "Yes/No", the model predicts the *actual number* of reallocated sectors.
* **Surface Degradation:** By predicting a rise in SMART 5, we can intervene before the disk's internal spare area is fully depleted.
* **Failure Urgency:** A higher predicted SMART 5 value correlates directly with imminent mechanical failure.

![regression.png](Graphs/regression.png)

---

### Top Predictors (Feature Importance):
The following SMART attributes were identified as the strongest indicators of failure:
1. **SMART 5** (Reallocated Sectors Count)
2. **SMART 187** (Reported Uncorrectable Errors)
3. **SMART 188** (Command Timeout)
4. **SMART 197** (Current Pending Sector Count)

---

## Clustering Analysis

Using the **K-Means** algorithm and **t-SNE** visualization (Euclidean distance), drives are categorized into 3 distinct groups:

* **Cluster 0:** Healthy drives (Optimal operation).
* **Cluster 1:** Aging drives (Increased power-on hours/usage).
* **Cluster 2:** Critical drives (High probability of failure due to critical SMART errors).

![clustering.png](Graphs/clustering.png)

---

### Example - Graph Risk Profile Interpretation for most critical disk

To demonstrate the efficacy of our risk assessment, we present two extreme instances (hard drives) from the dataset. This visual comparison illustrates how various machine learning methods and two key attributes contribute to the final prediction.

![worst_case.png](Graphs/worst_case.png)

**High-Risk Instance (97%):** We observe a high level of convergence across all modules. Each machine learning method (classification, regression, and clustering) predicts values close to 1.0. This alignment confirms that the 97% risk score is highly accurate and reliable, as it is corroborated by multiple independent models simultaneously.

![best_case.png](Graphs/best_case.png)

**Low-Risk Instance (5%):** In contrast, we see a consistent prediction near 0.0 across all security-related modules. The only significant value is Age (0.8), proving that the model can effectively isolate natural wear and tear from actual failure signals.

**Formula for prediction calculation:** The HIR score is the weighted root-mean-square of four critical metrics ($K, R, G, N$), combining them into a single value that heavily penalizes large health deviations to predict imminent disk failure:

![HIR-png](Graphs/HIR.png)

---

## Chat interface

![img.png](Graphs/img.png)

---

## Branches

**main branch:** whole setup is running locally on my TrueNAS server via Portainer (LLM Llama 3).

**laptopVersion branch:** modified model and app.py for using better GPU and CPU of my laptop (LLM mistral-nemo:12b).