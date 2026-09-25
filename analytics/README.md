
# ZEPTO DATA & AI PLATFORM
## Analytics Pipeline

This module performs exploratory data analysis, machine-learning classification,
class imbalance handling, hyperparameter tuning, and a regression side-task
using the Titanic dataset.

## 1. Module Structure

analytics/
├── 01_eda.ipynb
├── 02_modeling.ipynb
├── titanic.csv
├── model_pipeline.joblib
└── README.md

## 2. Installation

Install the required Python libraries:

pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn joblib

The notebooks can be executed using Google Colab or Jupyter Notebook.

## 3. Execution Order

01_eda.ipynb
    ↓
titanic.csv
    ↓
02_modeling.ipynb
    ↓
model_pipeline.joblib

## 4. 01_eda.ipynb

The EDA notebook:

- Loads the Titanic dataset.
- Examines dataset structure and data types.
- Identifies missing values.
- Removes the deck column because of high missingness.
- Imputes missing numerical and categorical values.
- Examines distributions and outliers.
- Analyzes survival rates.
- Examines correlations.
- Demonstrates numerical standardization.
- Saves the cleaned dataset as titanic.csv.

## 5. 02_modeling.ipynb

The modeling notebook:

- Loads titanic.csv.
- Separates features and target.
- Performs a stratified train/test split.
- Applies preprocessing through pipelines.
- Trains Logistic Regression.
- Trains Decision Tree.
- Trains Random Forest.
- Handles class imbalance using SMOTE.
- Performs Random Forest hyperparameter tuning using GridSearchCV.
- Evaluates classification models.
- Performs a fare-prediction regression task.
- Saves the final classification pipeline as model_pipeline.joblib.

## 6. Classification Models

The following models were evaluated:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Random Forest + SMOTE
5. Tuned Random Forest

Evaluation metrics:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

Random Forest was also evaluated using out-of-bag scoring.

## 7. Classification Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8045 | 0.7931 | 0.6667 | 0.7244 | 0.8439 |
| Decision Tree | 0.7654 | 0.7547 | 0.5797 | 0.6557 | 0.7971 |
| Random Forest | 0.8101 | 0.7778 | 0.7101 | 0.7424 | 0.8334 |
| Random Forest + SMOTE | 0.8101 | 0.7612 | 0.7391 | 0.7500 | 0.8415 |
| Tuned Random Forest | 0.7933 | 0.8200 | 0.5942 | 0.6891 | 0.8412 |

Random Forest OOB Score: 0.8020

Best GridSearchCV cross-validation ROC-AUC: 0.8780

## 8. Regression Side-Task

A Linear Regression model was used to predict passenger fare.

RMSE: 30.4731

R²: 0.3999

## 9. SMOTE

SMOTE was applied only to the training data through an imbalanced-learn
pipeline. The test set was kept unchanged.

Random Forest + SMOTE increased recall from 0.7101 to 0.7391 and F1-score
from 0.7424 to 0.7500 compared with the baseline Random Forest.

## 10. Hyperparameter Tuning

Random Forest hyperparameters were tuned using GridSearchCV with 5-fold
cross-validation and ROC-AUC as the scoring metric.

Best parameters:

max_depth = 5
min_samples_leaf = 1
min_samples_split = 2
n_estimators = 200

## 11. Saved Model

The final classification pipeline is saved as:

model_pipeline.joblib

The saved pipeline contains preprocessing, SMOTE, and the Random Forest
classifier.

The saved pipeline was reloaded and successfully tested on sample data.

## 12. Design Decisions

- A stratified train/test split was used to preserve target-class proportions.
- Numerical variables were standardized.
- Categorical variables were one-hot encoded.
- Preprocessing was placed inside pipelines.
- SMOTE was applied only to training data.
- Multiple classification metrics were used rather than accuracy alone.
- GridSearchCV was used for Random Forest hyperparameter tuning.
- The final saved model contains the complete preprocessing and classification
  pipeline.

## 13. Reproducibility

Random states were fixed where applicable.

Recommended execution order:

01_eda.ipynb
    ↓
titanic.csv
    ↓
02_modeling.ipynb
    ↓
model_pipeline.joblib
