# Machine Learning - Hard Drive Failure Prediction

A machine learning project that predicts hard drive failures using historical SMART sensor data.
The model learns failure patterns over time to identify disks at risk before they break.

## Project Overview
This project aims to develop a predictive maintenance pipeline for data storage reliability. Using Large-scale SMART (Self-Monitoring, Analysis, and Reporting Technology) data, the goal is to identify early warning signs of hardware failure to prevent data loss and optimize replacement cycles.

The project utilizes the **Backblaze** open-source dataset, processing millions of daily disk snapshots to build a robust classification and regression system.

## Current Development Status
The project is currently focused on **Data Engineering and Preprocessing**. Due to the high volume of data (30M+ records per month), the following optimizations have been implemented:

* **Failure-Centric Extraction:** Optimized search across quarterly datasets (Q1–Q4) to isolate all hardware failure events (`failure=1`).
* **Feature Selection:** Dimensionality reduction focused on 19 critical SMART attributes identified as statistically significant for failure prediction.
* **Historical Consolidation:** Aggregation of annual failure instances into a master dataset for improved model generalization.

## Technical Roadmap
The project will evolve through the following stages:

1.  **Clustering:** Unsupervised learning to group drive instances based on usage patterns and hardware specifications.
2.  **Imbalance Correction (SMOTE):** Implementing *Synthetic Minority Over-sampling Technique* to address the extreme class imbalance between healthy and failed drives.
3.  **RUL Regression:** Predicting the *Remaining Useful Life* (RUL) of a disk based on degradation trends.
4.  **Ensemble Classification (Feature Stacking):** Utilizing a stacked model architecture to provide final probability scores for potential failures.



## Project Goal
The objective is to move from reactive maintenance to a proactive strategy. By leveraging historical SMART data, this project intends to demonstrate how machine learning can provide a reliable "early warning system" for large-scale data center infrastructure.

---
*Status: Active Development*
