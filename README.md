# Red Wine Quality — Finding the Best Classifier

## Overview
Classification project to predict red wine quality using machine learning.
Compared three classifiers to find the best performing model on a 
real-world imbalanced dataset of 1,599 wines.

## Problem Statement
Red wine quality is rated on a scale of 3–9 based on 11 physicochemical 
properties (alcohol, acidity, pH, sulphates etc.). The goal was to build 
a model that accurately predicts wine quality and identify which 
classifier performs best.

## Key Challenge — Class Imbalance
Most wines in the dataset were rated average (5–6). Very few records 
had extreme ratings (3 or 9). Training a model on this imbalanced data 
would bias predictions toward average quality.

Solution: Used SMOTE (Synthetic Minority Oversampling Technique) to 
synthetically balance the dataset before training — ensuring the model 
learns to distinguish all quality levels fairly.

## Results

| Classifier | Accuracy |
|---|---|
| Logistic Regression | 58.7% |
| Decision Tree | 79.2% |
| Random Forest | **85.9%** ✅ |

## Why Random Forest Won
Wine quality depends on many features interacting together — not a 
single rule. Logistic Regression draws a simple boundary which cannot 
capture this complexity. Random Forest builds hundreds of decision trees, 
each considering different feature combinations, then votes on the final 
prediction — making it far more powerful for complex datasets.

## Tools and Libraries
- Python
- pandas, numpy
- scikit-learn (LogisticRegression, DecisionTreeClassifier, RandomForestClassifier)
- imbalanced-learn (SMOTE)
- matplotlib, seaborn

## Key Skills Demonstrated
- Handling imbalanced datasets with SMOTE
- Feature scaling with StandardScaler
- Comparing multiple ML classifiers
- Model evaluation with accuracy score

## Dataset
[Red Wine Quality Dataset](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009) 
— UCI Machine Learning Repository via Kaggle
