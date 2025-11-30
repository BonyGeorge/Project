# Student GPA Prediction Analysis (HSLS:09)

## Project Overview
This project utilizes machine learning techniques to predict student Grade Point Average (GPA) categories based on demographic, academic, and behavioral data from the High School Longitudinal Study of 2009 (HSLS:09). The notebook demonstrates a full end-to-end data science pipeline, including data cleaning, advanced feature selection, dimensionality reduction, and ensemble modeling.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Files Description](#files-description)
3. [Methodology](#methodology)
4. [Models Benchmarked](#models-benchmarked)
5. [Key Findings](#key-findings)

## Prerequisites
To run this notebook, you will need the following Python libraries installed:

```shell 
pip install pandas numpy matplotlib scikit-learn statsmodels imbalanced-learn xgboost

## Files Description
* **`Modeling.ipynb`**: The main Jupyter Notebook containing the analysis, preprocessing, and modeling code.
* **`data/our_features_cleaned.csv`**: The cleaned dataset used as input (required for the notebook to run).
* **`resources/student_info_updated.json`**: Metadata file mapping features to categorical, numerical, or boolean types.

## Methodology

The analysis follows a structured pipeline:

### 1. Data Preparation
* **Loading:** Data is loaded and the target variable `X5GPAALL` is isolated.
* **Cleaning:** Features with high missing value counts (e.g., suspension records) are dropped.
* **Imputation:** Missing numerical values are imputed with the mean; categorical values with the mode.
* **Scaling:** `StandardScaler` is applied to normalize feature magnitudes.

### 2. Handling Class Imbalance
* **SMOTE:** The Synthetic Minority Over-sampling Technique is applied to the training set to address imbalances in GPA categories, ensuring the model does not bias toward the majority class.

### 3. Feature Engineering & Selection
* **Backward Elimination:** A wrapper method using OLS (MNLogit) logic is used to iteratively remove features with p-values > 0.05 to reduce noise.
* **Dimensionality Reduction:** Three techniques were compared to reduce the feature space:
    * **PCA** (Principal Component Analysis)
    * **LDA** (Linear Discriminant Analysis) - *Selected as the best performer.*
    * **Kernel PCA** (Non-linear reduction)

### 4. Evaluation Strategy
* Models are evaluated using **Precision**, **Recall**, **F1-Score (Macro)**, and **Accuracy**.
* A final comparison plot visualizes these metrics across all trained models.

## Models Benchmarked
The following algorithms were trained and tuned (using `GridSearchCV` where applicable):

1.  **Logistic Regression** (Baseline & with Reduced Features)
2.  **Stochastic Gradient Descent (SGD)**
3.  **XGBoost Classifier**
4.  **Random Forest Classifier**
5.  **Support Vector Classifier (SVC)** (RBF Kernel)
6.  **AdaBoost Classifier**
7.  **Gaussian Naive Bayes**
8.  **Voting Classifier** (Ensemble of XGBoost, RF, SVC, and Naive Bayes)

## Key Findings
* **Feature Importance:** Academic predictors (like Math scores) and behavioral indicators proved statistically significant during backward elimination.
* **Dimensionality Reduction:** LDA provided better class separability than PCA for this specific dataset.
* **Best Model:** (Based on the notebook outputs) The **AdaBoost Classifier** and **Support Vector Classifier (SVC)** showed the strongest performance metrics, achieving an F1-Score and Accuracy of approximately **43-44%** on the test set.

---
*Created by Group 2.*