a. Problem statement: Implement and compare 6 classification models to predict income levels (<=50K or >50K).

b. Dataset description [ 1 mark ]: Input dataset consists of demographics of individuals along with their income classified as >50K or <=50K. This is a classic binary classification problem used to understand socioeconomic factors affecting income levels. It has applications in economic policy, social services allocation, and understanding income inequality patterns. The dataset consists of 15 features (list given below) and has few columns with missing values which need to handled.

| #   Column         | Data Type |
|--------------------|-----------|
| 0   age            | int64     |
| 1   workclass      | object    |
| 2   fnlwgt         | int64     |
| 3   education      | object    |
| 4   education-num  | int64     |
| 5   marital-status | object    |
| 6   occupation     | object    |
| 7   relationship   | object    |
| 8   race           | object    |
| 9   sex            | object    |
| 10  capital-gain   | int64     |
| 11  capital-loss   | int64     |
| 12  hours-per-week | int64     |
| 13  native-country | object    |
| 14  income         | object    |

c. Models used: [ 6 marks - 1 marks for all the metrics for each model ] Make a Comparison Table with the evaluation metrics calculated for all the 6 models as below:
|   | Model                  | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
|---|------------------------|----------|--------|-----------|--------|--------|--------|
| 0 | Logistic Regression    | 0.8177   | 0.85   | 0.8062    | 0.8177 | 0.8019 | 0.4617 |
| 1 | Decision Tree          | 0.8066   | 0.7404 | 0.8062    | 0.8066 | 0.8064 | 0.4817 |
| 2 | K-Nearest Neighbors    | 0.8253   | 0.8433 | 0.8191    | 0.8253 | 0.8212 | 0.5142 |
| 3 | Naive Bayes (Gaussian) | 0.7978   | 0.8498 | 0.783     | 0.7978 | 0.7697 | 0.3798 |
| 4 | Random Forest          | 0.8541   | 0.9027 | 0.8489    | 0.8541 | 0.85   | 0.5927 |
| 5 | XGBoost                | 0.8624   | 0.9227 | 0.8576    | 0.8624 | 0.8583 | 0.6154 |

- Add your observations on the performance of each model on the chosen dataset. [ 3 marks ]

1. OVERALL PERFORMANCE RANKING (by Accuracy):
--------------------------------------------------------------------------------
1. XGBOOST CLASSIFIER: 0.8624
2. RANDOM FOREST CLASSIFIER: 0.8541
3. K-NEAREST NEIGHBORS CLASSIFIER: 0.8253
4. LOGISTIC REGRESSION: 0.8177
5. DECISION TREE CLASSIFIER: 0.8066
6. NAIVE BAYES CLASSIFIER (GAUSSIAN): 0.7978

2. DETAILED OBSERVATIONS:
--------------------------------------------------------------------------------

LOGISTIC REGRESSION:
  ✓ Good performance with 81.77% accuracy
  ✓ Good discrimination capability (AUC: 0.8500)
  ✓ Well-balanced precision (0.8062) and recall (0.8177)
  • F1-Score: 0.8019 - Overall balance between precision and recall
  ○ Moderate correlation (MCC: 0.4617)

DECISION TREE CLASSIFIER:
  ✓ Good performance with 80.66% accuracy
  ○ Moderate discrimination capability (AUC: 0.7404)
  ✓ Well-balanced precision (0.8062) and recall (0.8066)
  • F1-Score: 0.8064 - Overall balance between precision and recall
  ○ Moderate correlation (MCC: 0.4817)

K-NEAREST NEIGHBORS CLASSIFIER:
  ✓ Good performance with 82.53% accuracy
  ✓ Good discrimination capability (AUC: 0.8433)
  ✓ Well-balanced precision (0.8191) and recall (0.8253)
  • F1-Score: 0.8212 - Overall balance between precision and recall
  ✓ Strong correlation (MCC: 0.5142)

NAIVE BAYES CLASSIFIER (GAUSSIAN):
  ○ Moderate performance with 79.78% accuracy
  ✓ Good discrimination capability (AUC: 0.8498)
  ✓ Well-balanced precision (0.7830) and recall (0.7978)
  • F1-Score: 0.7697 - Overall balance between precision and recall
  ○ Moderate correlation (MCC: 0.3798)

RANDOM FOREST CLASSIFIER:
  ✓ Excellent performance with 85.41% accuracy
  ✓ Excellent discrimination capability (AUC: 0.9027)
  ✓ Well-balanced precision (0.8489) and recall (0.8541)
  • F1-Score: 0.8500 - Overall balance between precision and recall
  ✓ Strong correlation (MCC: 0.5927)

XGBOOST CLASSIFIER:
  ✓ Excellent performance with 86.24% accuracy
  ✓ Excellent discrimination capability (AUC: 0.9227)
  ✓ Well-balanced precision (0.8576) and recall (0.8624)
  • F1-Score: 0.8583 - Overall balance between precision and recall
  ✓ Strong correlation (MCC: 0.6154)

3. KEY INSIGHTS:
--------------------------------------------------------------------------------

• Best Overall Model: XGBOOST CLASSIFIER
  - Achieved highest accuracy of 86.24%
  - F1-Score: 0.8583

• Best AUC Score: XGBOOST CLASSIFIER (0.9227)

• Best MCC Score: XGBOOST CLASSIFIER (0.6154)

• Ensemble Models Average Accuracy: 85.83%
• Traditional Models Average Accuracy: 81.18%
  → Ensemble models outperform traditional models by 5.7%

• Lowest Performing Model: NAIVE BAYES CLASSIFIER (GAUSSIAN) (79.78%)

4. RECOMMENDATIONS:
--------------------------------------------------------------------------------

• For deployment: Use XGBOOST CLASSIFIER for best overall performance
• For interpretability: Consider DECISION TREE or LOGISTIC REGRESSION
• For handling imbalanced data: Consider models with high MCC scores
• The XGBOOST CLASSIFIER shows excellent ability to distinguish between classes

================================================================================

