# Customer Churn Analysis

A machine learning project to predict customer churn in the telecom sector using **Logistic Regression**, **Random Forest**, and **Support Vector Machine (SVM)**.
---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [How to Run](#how-to-run)
- [Models Used](#models-used)
- [Results](#results)
- [Visualizations](#visualizations)
- [Future Improvements](#future-improvements)
- [Team](#team)

---

## Overview

Customer churn — when a customer stops using a service — is a critical problem for telecom companies. Retaining existing customers is significantly cheaper than acquiring new ones. This project builds a binary classification pipeline that predicts whether a customer will churn based on their usage patterns and account information.

**Key goals:**
- Clean and preprocess real-world telecom data
- Train and compare three ML models
- Evaluate using multiple metrics (Accuracy, F1, ROC-AUC, Log-Loss)
- Visualize EDA findings and model performance

---

## Dataset

**Source:** [Telco Customer Churn — Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn)

| Property | Value |
|---|---|
| Records | 7,043 customers |
| Features | 19 (numerical + categorical) |
| Target | `Churn` (1 = churned, 0 = stayed) |
| Churn Rate | ~26.5% |

**Key features include:** tenure, MonthlyCharges, TotalCharges, Contract type, InternetService, PaymentMethod, SeniorCitizen, and more.

---

## Project Structure

```
churn-prediction/
│
├── churn_prediction.py          # Main script — full ML pipeline
├── model_summary.csv            # Model metrics comparison table
│
├── plots/
│   ├── 01_churn_distribution.png
│   ├── 02_boxplots.png
│   ├── 03_correlation_heatmap.png
│   ├── 04_confusion_matrices.png
│   ├── 05_roc_curve_comparison.png
│   ├── 06_learning_curve_accuracy.png
│   ├── 07_learning_curve_logloss.png
│   ├── 08_feature_importance_lr.png
│   ├── 09_feature_importance_rf.png
│   ├── 10_scatter_feature_importance.png
│   └── 11_model_summary_table.png
│
└── README.md
```

---

## Tech Stack

| Library | Purpose |
|---|---|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `matplotlib` | Plotting and visualizations |
| `seaborn` | Statistical visualizations |
| `scikit-learn` | ML models, preprocessing, evaluation |

---

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/your-username/churn-prediction.git
cd churn-prediction
```

### 2. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 3. Download the dataset
Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` from [Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn) and place it in the project root.

### 4. Update the script
In `churn_prediction.py`, replace the synthetic data section with:
```python
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
```

### 5. Run
```bash
python churn_prediction.py
```

All plots will be saved to the `output/` directory.

---

## Models Used

### Logistic Regression
A linear binary classifier that models churn probability using a sigmoid function. Simple, interpretable, and fast — but assumes linearity.

### Random Forest
An ensemble of decision trees trained on random data subsets. Handles non-linear relationships well and is robust to overfitting.

### Support Vector Machine (SVM)
Finds the optimal hyperplane to separate churn vs. non-churn customers. Uses an RBF kernel for non-linear classification.

---

## Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Log-Loss |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7864 | 0.6289 | 0.4867 | 0.5487 | 0.8148 | 0.4527 |
| Random Forest | 0.7693 | 0.5941 | 0.4282 | 0.4977 | 0.8007 | 0.4840 |
| SVM | 0.7779 | 0.6061 | 0.4787 | 0.5349 | 0.8048 | 0.4659 |

**Key takeaways:**
- **Logistic Regression** achieved the highest ROC-AUC (0.8148) and lowest log-loss
- **SVM** showed competitive performance with good class separation
- **Random Forest** had strong recall, identifying more actual churn cases
- All models struggled with the class imbalance (~73% non-churn vs ~27% churn)

---

## Visualizations

| Plot | Description |
|---|---|
| Churn Distribution | Class balance bar chart + churn by contract type |
| Box Plots | Feature distributions split by churn label |
| Correlation Heatmap | Pairwise correlations among numerical features |
| Confusion Matrices | Side-by-side for all three models |
| ROC Curve Comparison | AUC comparison across models |
| Learning Curves | Accuracy and log-loss vs. training size |
| Feature Importance | LR coefficients and RF Gini importance (top 15 features) |

---

## Future Improvements

- **Handle class imbalance** using SMOTE or class-weight adjustments
- **Hyperparameter tuning** with GridSearchCV or RandomizedSearchCV
- **Ensemble methods** — Voting Classifier or Stacking
- **Deep learning** — experiment with a simple neural network
- **Model deployment** — Flask/FastAPI REST API for real-time predictions
- **Feature engineering** — time-based features from tenure data

---

## License

This project is for academic purposes only.
