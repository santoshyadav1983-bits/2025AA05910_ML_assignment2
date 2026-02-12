a. Problem statement: Implement and compare 6 classification models to predict income levels (<=50K or >50K).
b. Dataset description [ 1 mark ]: Input dataset consists of demographics on individuals along with their income classified as >50K or <=50K. This is a classic binary classification problem used to understand socioeconomic factors affecting income levels. It has applications in economic policy, social services allocation, and understanding income inequality patterns. The dataset consists of 15 features (list given below) and has few columns with missing values which need to handled.

Column	Dtype
age	int64
workclass	object
fnlwgt	int64
education	object
education-num	int64
marital-status	object
occupation	object
relationship	object
race	object
sex	object
capital-gain	int64
capital-loss	int64
hours-per-week	int64
native-country	object
income	object

c. Models used: [ 6 marks - 1 marks for all the metrics for each model ] Make a Comparison Table with the evaluation metrics calculated for all the 6 models as below:
ML Model Name
Accuracy
AUC
Precision
Recall
F1
MCC
Logistic Regression
Decision Tree
kNN
Naive Bayes
Random Forest (Ensemble)
XGBoost (Ensemble)
- Add your observations on the performance of each model on the chosen dataset. [ 3 marks ]
ML Model Name
Observation about model performance
Logistic Regression
Decision Tree
kNN
Naive Bayes
Random Forest (Ensemble)
XGBoost (Ensemble)
