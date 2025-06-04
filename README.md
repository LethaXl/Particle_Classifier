# Particle Classifier

This project applies data mining techniques to classify high-energy particle events using the [MAGIC Gamma Telescope dataset](https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope). The goal is to distinguish between gamma-ray signals and hadronic background events.

## 🔍 Overview
We evaluate multiple classification algorithms, including:
- Decision Trees
- Naive Bayes
- Support Vector Machines (RBF)
- Random Forest
- XGBoost

The analysis includes:
- Exploratory Data Analysis (EDA)
- Data Preprocessing
- Feature Selection (F-Score and RFE)
- Model Evaluation using Accuracy, Precision, Recall, AUC
- Visualization of Confusion Matrices and ROC Curves

## 🧠 Dataset
The dataset is sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope). It contains 19,020 simulated particle events, each with 10 real-valued features and a binary label (gamma or hadron).

## 📈 Results
XGBoost and Random Forest performed best, achieving:
- **XGBoost**: 88.47% accuracy, AUC = 0.9381
- **Random Forest**: 88.33% accuracy, AUC = 0.9351

Feature selection reduced model complexity while maintaining strong classification performance.

## 📁 Files
- `Project_Code.py` – Full pipeline with model training, evaluation, and feature selection
- `magic_gamma_dataset.data` – Raw dataset used for classification
- `Particle_Classification_Report.pdf` – Full report with analysis, graphs, and discussion
