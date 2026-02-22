# Machine Learning - Hard Drive Failure Prediction Model & AI Assistant

A machine learning project that predicts hard drive failures using historical **SMART** (Self-Monitoring, Analysis, and Reporting Technology) sensor data. This system integrates Random Forest classification, regression, unsupervised clustering, and a local AI assistant for result interpretation. 

By learning disk degradation patterns, the model identifies risks before critical hardware failure and data loss occur.

### Key Features:
* **Dataset:** Backblaze open-source data (Year 2025).
* **Scale:** Processed 32M+ records, filtered into a balanced dataset of **8,828 instances**.
* **Methodology:** Supervised learning (Random Forest - regression and classification) and Unsupervised learning (K-Means).
* **AI Integration:** Local LLM (Llama 3 via Ollama) providing natural language explanations for SMART parameters.

---

## Technical Architecture
The project is deployed in an isolated **Docker** environment on **TrueNAS SCALE**, ensuring data privacy and system stability.

### System Components:
1.  **ML Model:** Random Forest Classifier trained on 19 statistically significant SMART attributes.
2.  **Streamlit UI:** A web dashboard for AI chat interaction.
3.  **Ollama Service:** Local inference engine running the Llama 3 model (mistral-nemo:12b on other branch, meant for laptop).
4.  **Data Pipeline:** Automated preprocessing, median imputation, and feature scaling.

---

## Model Performance & Analysis

The model demonstrates high reliability in identifying failure-prone drives using dual-method analysis:

### 1. Classification Analysis (Yes/No)
Categorizes drives into binary states: **Healthy** or **Failure-Prone**.
* **Accuracy:** 90.15%
* **Recall:** 86.00% (Critical for capturing actual failure events)
* **F1-Score:** 0.88

### 2. Regression Analysis (SMART 5)
Predicting the value of **SMART 5 (Reallocated Sectors Count)**.
* **Predictive Forecasting:** Instead of a binary "Yes/No", the model predicts the *actual number* of reallocated sectors.
* **Surface Degradation:** By predicting a rise in SMART 5, we can intervene before the disk's internal spare area is fully depleted.
* **Failure Urgency:** A higher predicted SMART 5 value correlates directly with imminent mechanical failure.

---

### Top Predictors (Feature Importance):
The following SMART attributes were identified as the strongest indicators of failure:
1.  **SMART 5** (Reallocated Sectors Count)
2.  **SMART 187** (Reported Uncorrectable Errors)
3.  **SMART 188** (Command Timeout)
4.  **SMART 197** (Current Pending Sector Count)

---

## Clustering Analysis
Using the **K-Means** algorithm and **t-SNE** visualization (Euclidean distance), drives are categorized into 3 distinct groups:
* **Cluster 0:** Healthy drives (Optimal operation).
* **Cluster 1:** Aging drives (Increased power-on hours/usage).
* **Cluster 2:** Critical drives (High probability of failure due to critical SMART errors).

---

## Chat interface

![img.png](img.png)

---

main branch : whole setup is running localy on my treunas server via portainer (LLM llama3)

laptopVersion branch : modified model and app.py for using better gpu and cpu of my laptop (LLM mistral-nemo:12b)
