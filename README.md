
# Particle Classifier

A machine learning project focused on classifying high-energy particle events using the [MAGIC Gamma Telescope dataset](https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope). The objective is to distinguish gamma-ray signals from hadronic background events by comparing classification algorithms, exploring feature selection techniques and evaluating predictive performance.

## Overview

This project explores supervised machine learning techniques for binary classification. Five algorithms were evaluated to compare their effectiveness in identifying gamma-ray events:

- Decision Trees
- Naive Bayes
- Support Vector Machines (RBF)
- Random Forest
- XGBoost

The analysis covers exploratory data analysis (EDA), data preprocessing, feature selection, model evaluation and performance visualization.

## Dataset

The dataset was obtained from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope).

| Property | Description |
| :--- | :--- |
| Dataset | MAGIC Gamma Telescope |
| Instances | 19,020 |
| Features | 10 numerical attributes |
| Target | Gamma-ray or hadron |
| Task | Binary classification |

The features describe characteristics of particle showers recorded by the telescope. These measurements are used to identify patterns that distinguish gamma-ray events from background hadronic events.

## Methodology

### 1. Exploratory Data Analysis

Explored the dataset to understand feature distributions, relationships between variables and differences between the two target classes.

### 2. Data Preprocessing

Prepared the dataset for classification and examined the characteristics of the input features before model evaluation.

### 3. Feature Selection

Applied two feature selection techniques:

- **F-Score:** Evaluated the relevance of individual features to the classification target.
- **Recursive Feature Elimination (RFE):** Examined feature subsets by iteratively removing less important features.

These techniques were used to investigate whether reducing the number of input features could maintain strong classification performance.

### 4. Model Evaluation

Compared the classification algorithms using several evaluation metrics:

| Metric | Purpose |
| :--- | :--- |
| Accuracy | Measures the proportion of correctly classified events |
| Precision | Measures the reliability of positive predictions |
| Recall | Measures the proportion of actual positive events identified |
| AUC | Evaluates how well a model distinguishes between the two classes |

Confusion matrices and ROC curves were also used to visualize classification performance.

## Results

XGBoost and Random Forest achieved the strongest overall results among the evaluated models.

| Model | Accuracy | AUC |
| :--- | ---: | ---: |
| XGBoost | 88.47% | 0.9381 |
| Random Forest | 88.33% | 0.9351 |

Both models demonstrated strong classification performance, with XGBoost achieving slightly higher accuracy and AUC.

Feature selection also showed that reducing the number of input features could maintain competitive predictive performance while simplifying the models.

## Key Findings

- **Model Performance:** XGBoost achieved the highest reported accuracy and AUC, closely followed by Random Forest.
- **Feature Selection:** F-Score and RFE helped explore the importance of individual features and the impact of using smaller feature subsets.
- **Evaluation:** Comparing multiple metrics provided a more comprehensive understanding of model performance than accuracy alone.
- **Visualization:** Confusion matrices and ROC curves helped illustrate classification behaviour and differences between algorithms.

## Conclusion

This project provided practical experience applying machine learning techniques to a real-world scientific classification problem. By comparing five algorithms, exploring feature selection methods and analyzing evaluation metrics, I gained a better understanding of how different models perform on structured numerical data.

The results highlighted the effectiveness of ensemble learning methods, particularly XGBoost and Random Forest, for distinguishing gamma-ray signals from hadronic background events.

## Files

- `Project_Code.py` - Python implementation of the classification pipeline, including model training, evaluation and feature selection.
- `magic_gamma_dataset.data` - Dataset used for classification.
- [Particle Classification Report (PDF)](https://aseef-portfolio.vercel.app/Particle_Classification_Report.pdf) - Detailed analysis, visualizations, experimental results and discussion.
