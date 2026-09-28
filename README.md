# 🛡️ NewsGuard — AI-Powered Fake News Detection System

NewsGuard is an AI/ML-based fake news detection system that analyzes the linguistic patterns of a news statement and predicts whether it is **REAL** or **FAKE**.

The system combines **TF-IDF text representation, Machine Learning classification, and SHAP explainability** with a Flask-based web dashboard.

---

## 🎯 Project Objective

The objective of NewsGuard is to help users identify potentially misleading or unreliable news content by analyzing the text using machine learning.

The system provides:

- Fake/Real prediction
- Prediction confidence
- Fake probability
- Real probability
- Explainable AI signals
- Interactive web dashboard
- Flask REST API

> **Note:** NewsGuard is an AI-assisted detection tool and should not be treated as a definitive fact-checking service.

---

## ✨ Features

### 📰 Fake News Detection
Users can enter a news statement into the dashboard and receive an AI-based prediction.

### 📊 Probability Analysis
The system displays:

- FAKE probability
- REAL probability
- Prediction confidence

### 🔎 Explainable AI

NewsGuard uses **SHAP (SHapley Additive exPlanations)** to identify text features that influenced the model prediction.

Example:

```text
cancer       → FAKE
scientists   → FAKE
doctors      → FAKE
hours        → REAL