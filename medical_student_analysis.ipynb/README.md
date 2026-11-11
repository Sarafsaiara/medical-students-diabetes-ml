# Medical Students Health & Diabetes Prediction Project

An end-to-end Machine Learning pipeline built for an undergraduate Artificial Intelligence course to analyze student health metrics, handle severe class imbalance using SMOTE, and predict the likelihood of diabetes using multiple classification models.

---

##  Project Overview
Healthcare datasets often suffer from missing values and class imbalance. This project investigates the **Medical Students Dataset** (approx. 200,000 records) to preprocess physiological indicators, engineer categorical features, balance the target distribution, and evaluate various supervised machine learning algorithms.

---

##  Dataset & Features
* **Source:** Medical Students Dataset (`medical_students_dataset.csv`)[cite: 1]
* **Target Variable:** `Diabetes` (Binary: Yes/No)[cite: 1]
* **Numerical Features:** Age, Height, Weight, BMI, Temperature, Heart Rate, Blood Pressure, Cholesterol[cite: 1]
* **Categorical / Binary Features:** Gender, Blood Type, Smoking Status[cite: 1]

---

##  Methodology & Pipeline

1. **Data Cleaning & Imputation:** 
   * Dropped irrelevant identifiers (`Student ID`)[cite: 1].
   * Imputed missing values in numerical columns with respective column means[cite: 1].
   * Imputed missing categorical values with modes and applied one-hot encoding[cite: 1].
2. **Handling Class Imbalance (SMOTE):** 
   * Addressed the ~10:1 class imbalance in the target variable (`Diabetes`) using **Synthetic Minority Over-sampling Technique (SMOTE)** to ensure balanced training data[cite: 1].
3. **Model Training & Evaluation:** 
   * Trained and compared four classifiers: **Random Forest**, **Logistic Regression**, **K-Nearest Neighbors (KNN)**, and **XGBoost**[cite: 1].
   * Evaluated performance using ROC-AUC curves and feature importance rankings[cite: 1].

---

##  Key Results & Visualizations
* **Top Predictive Features:** Random Forest feature importance highlighted **Age**, **Cholesterol**, **Heart Rate**, and **Blood Pressure** as leading indicators[cite: 1].
* **Model Comparison:** Evaluated via ROC-AUC metrics to balance True Positive and False Positive rates across imbalanced thresholds[cite: 1].


