# Machine Failure Prediction System

This project implements a machine learning based predictive maintenance system to detect potential machine failures using operational sensor data.

## Problem Statement
Unexpected machine failures lead to production downtime and high maintenance costs. This project predicts machine failure in advance to support preventive maintenance decisions.

## Dataset
AI4I Predictive Maintenance Dataset

## Features Used
- Air temperature [K]
- Process temperature [K]
- Rotational speed [rpm]
- Torque [Nm]
- Tool wear [min]
- Machine type (L, M, H)

## Models Implemented
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting (Final Model)

## Evaluation Metric
- F1 Score (chosen due to class imbalance)

## Deployment
The final Gradient Boosting model is deployed using Streamlit to allow real-time machine failure prediction.

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Streamlit
- Joblib

## How to Run
streamlit run app.py


## Author
Dhananjay


