# Machine Learning - Hard Drive Failure Prediction & AI Assistant

A machine learning project that predicts hard drive failures using historical **SMART** (Self-Monitoring, Analysis, and Reporting Technology) sensor data. This system integrates Random Forest classification, unsupervised clustering, and a local AI assistant for real-time result interpretation.

## Project Overview
The objective is to move from reactive maintenance to a proactive strategy. By learning disk degradation patterns, the model identifies risks before critical hardware failure and data loss occur.

### Key Features:
* **Dataset:** Backblaze open-source data (Year 2025).
* **Scale:** Processed 32M+ records, filtered into a balanced dataset of **8,828 instances**.
* **Methodology:** Supervised learning (Random Forest) and Unsupervised learning (K-Means).
* **AI Integration:** Local LLM (Llama 3 via Ollama) providing natural language explanations for SMART parameters.

---

## Technical Architecture
The project is deployed in an isolated **Docker** environment on **TrueNAS SCALE**, ensuring data privacy and system stability.

### System Components:
1.  **ML Model:** Random Forest Classifier trained on 19 statistically significant SMART attributes.
2.  **Streamlit UI:** A web dashboard for cluster visualization (t-SNE) and AI chat interaction.
3.  **Ollama Service:** Local inference engine running the Llama 3 model.
4.  **Data Pipeline:** Automated preprocessing, median imputation, and feature scaling.



---

## Model Performance
The model demonstrates high reliability in identifying failure-prone drives:

* **Accuracy:** 90.15%
* **Recall:** 86.00% (Critical for capturing actual failure events)
* **F1-Score:** 0.88

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
