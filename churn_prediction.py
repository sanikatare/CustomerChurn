"""
Churn Prediction in Telecom Industry
Full ML Pipeline: EDA, Preprocessing, Modeling, Evaluation
Models: Logistic Regression, Random Forest, SVM
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve,
                             log_loss, f1_score, precision_score, recall_score)
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 0. Output folder
# ─────────────────────────────────────────────
import os
OUT = "/home/claude/output"
os.makedirs(OUT, exist_ok=True)

# ─────────────────────────────────────────────
# 1. Generate Synthetic Telco Dataset
# ─────────────────────────────────────────────
np.random.seed(42)
N = 7043

churn = np.random.choice([0, 1], size=N, p=[0.73, 0.27])

df = pd.DataFrame({
    "customerID":       [f"CUST{i:05d}" for i in range(N)],
    "gender":           np.random.choice(["Male", "Female"], N),
    "SeniorCitizen":    np.random.choice([0, 1], N, p=[0.84, 0.16]),
    "Partner":          np.random.choice(["Yes", "No"], N),
    "Dependents":       np.random.choice(["Yes", "No"], N, p=[0.3, 0.7]),
    "tenure":           np.clip(np.where(churn == 1,
                                         np.random.exponential(15, N),
                                         np.random.exponential(40, N)).astype(int), 0, 72),
    "PhoneService":     np.random.choice(["Yes", "No"], N, p=[0.9, 0.1]),
    "MultipleLines":    np.random.choice(["Yes", "No", "No phone service"], N, p=[0.42, 0.48, 0.10]),
    "InternetService":  np.random.choice(["DSL", "Fiber optic", "No"], N, p=[0.34, 0.44, 0.22]),
    "OnlineSecurity":   np.random.choice(["Yes", "No", "No internet service"], N, p=[0.28, 0.50, 0.22]),
    "TechSupport":      np.random.choice(["Yes", "No", "No internet service"], N, p=[0.29, 0.49, 0.22]),
    "StreamingTV":      np.random.choice(["Yes", "No", "No internet service"], N, p=[0.38, 0.40, 0.22]),
    "StreamingMovies":  np.random.choice(["Yes", "No", "No internet service"], N, p=[0.39, 0.39, 0.22]),
    "Contract":         np.where(churn == 1,
                                  np.random.choice(["Month-to-month", "One year", "Two year"], N, p=[0.80, 0.12, 0.08]),
                                  np.random.choice(["Month-to-month", "One year", "Two year"], N, p=[0.42, 0.28, 0.30])),
    "PaperlessBilling": np.random.choice(["Yes", "No"], N, p=[0.59, 0.41]),
    "PaymentMethod":    np.random.choice(["Electronic check", "Mailed check",
                                           "Bank transfer (automatic)", "Credit card (automatic)"], N),
    "MonthlyCharges":   np.clip(np.where(churn == 1,
                                          np.random.normal(74, 20, N),
                                          np.random.normal(61, 25, N)), 18, 118),
    "Churn":            churn
})
df["TotalCharges"] = (df["tenure"] * df["MonthlyCharges"] +
                      np.random.normal(0, 20, N)).clip(lower=18)

print("=" * 60)
print("CHURN PREDICTION IN TELECOM INDUSTRY")
print("=" * 60)
print(f"\nDataset shape: {df.shape}")
print(f"Churn rate   : {df['Churn'].mean() * 100:.1f}%")
print(f"\nData types:\n{df.dtypes.value_counts()}")

# ─────────────────────────────────────────────
# 2. EDA
# ─────────────────────────────────────────────
print("\n[EDA] Missing values:", df.isnull().sum().sum())
print(df.describe().round(2))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
df["Churn"].value_counts().plot.bar(ax=axes[0], color=["steelblue", "tomato"],
                                     edgecolor="black")
axes[0].set_title("Churn Distribution")
axes[0].set_xticklabels(["Not Churn", "Churn"], rotation=0)
axes[0].set_ylabel("Count")

pd.crosstab(df["Contract"], df["Churn"]).plot.bar(ax=axes[1], color=["steelblue", "tomato"],
                                                    edgecolor="black")
axes[1].set_title("Churn by Contract Type")
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=15, ha="right")
axes[1].legend(["Not Churn", "Churn"])
plt.tight_layout()
plt.savefig(f"{OUT}/01_churn_distribution.png", dpi=150)
plt.close()
print("  → Saved 01_churn_distribution.png")

# Box plot
num_cols = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
fig, axes = plt.subplots(1, 4, figsize=(16, 5))
for ax, col in zip(axes, num_cols):
    df.boxplot(column=col, by="Churn", ax=ax, patch_artist=True,
               boxprops=dict(facecolor="lightblue"))
    ax.set_title(col)
    ax.set_xlabel("Churn (0=No, 1=Yes)")
plt.suptitle("Box Plots of Numerical Features by Churn", y=1.01)
plt.tight_layout()
plt.savefig(f"{OUT}/02_boxplots.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → Saved 02_boxplots.png")

# Correlation heatmap (numerical only)
num_df = df[["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen", "Churn"]]
plt.figure(figsize=(7, 5))
sns.heatmap(num_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(f"{OUT}/03_correlation_heatmap.png", dpi=150)
plt.close()
print("  → Saved 03_correlation_heatmap.png")

# ─────────────────────────────────────────────
# 3. Preprocessing
# ─────────────────────────────────────────────
df_model = df.drop(columns=["customerID"])

# Label-encode binary categoricals
binary_cols = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]
le = LabelEncoder()
for col in binary_cols:
    df_model[col] = le.fit_transform(df_model[col])

# One-hot encode multi-category columns
cat_cols = ["MultipleLines", "InternetService", "OnlineSecurity",
            "TechSupport", "StreamingTV", "StreamingMovies",
            "Contract", "PaymentMethod"]
df_model = pd.get_dummies(df_model, columns=cat_cols, drop_first=True)

X = df_model.drop(columns=["Churn"])
y = df_model["Churn"]

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")

# ─────────────────────────────────────────────
# 4. Train Models
# ─────────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
    "Random Forest":        RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM":                  SVC(probability=True, kernel="rbf", random_state=42)
}

results = {}
print("\n" + "=" * 60)
print("MODEL EVALUATION RESULTS")
print("=" * 60)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred      = model.predict(X_test)
    y_prob      = model.predict_proba(X_test)[:, 1]

    acc   = accuracy_score(y_test, y_pred)
    prec  = precision_score(y_test, y_pred)
    rec   = recall_score(y_test, y_pred)
    f1    = f1_score(y_test, y_pred)
    auc   = roc_auc_score(y_test, y_prob)
    ll    = log_loss(y_test, y_prob)
    cm    = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    results[name] = dict(model=model, acc=acc, prec=prec, rec=rec,
                         f1=f1, auc=auc, ll=ll, cm=cm,
                         fpr=fpr, tpr=tpr, y_prob=y_prob)

    print(f"\n{'─'*40}")
    print(f"  {name}")
    print(f"{'─'*40}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {auc:.4f}")
    print(f"  Log-Loss  : {ll:.4f}")
    print(f"\n  Classification Report:\n{classification_report(y_test, y_pred, target_names=['Not Churn', 'Churn'])}")

# ─────────────────────────────────────────────
# 5. Confusion Matrices (separate plots)
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for ax, (name, r) in zip(axes, results.items()):
    sns.heatmap(r["cm"], annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Not Churn", "Churn"],
                yticklabels=["Not Churn", "Churn"])
    ax.set_title(f"Confusion Matrix\n{name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig(f"{OUT}/04_confusion_matrices.png", dpi=150)
plt.close()
print("\n  → Saved 04_confusion_matrices.png")

# ─────────────────────────────────────────────
# 6. ROC Curve Comparison
# ─────────────────────────────────────────────
plt.figure(figsize=(8, 6))
colors = ["royalblue", "darkorange", "green"]
for (name, r), color in zip(results.items(), colors):
    plt.plot(r["fpr"], r["tpr"], color=color,
             label=f"{name} (AUC = {r['auc']:.4f})", lw=2)
plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"{OUT}/05_roc_curve_comparison.png", dpi=150)
plt.close()
print("  → Saved 05_roc_curve_comparison.png")

# ─────────────────────────────────────────────
# 7. Learning Curves (Logistic Regression)
# ─────────────────────────────────────────────
lr_model = results["Logistic Regression"]["model"]
train_sizes, train_scores, val_scores = learning_curve(
    LogisticRegression(max_iter=500, random_state=42),
    X_train, y_train, cv=5, scoring="accuracy",
    train_sizes=np.linspace(0.1, 1.0, 10))

train_mean = train_scores.mean(axis=1)
val_mean   = val_scores.mean(axis=1)

plt.figure(figsize=(8, 5))
plt.plot(train_sizes, train_mean, "b-o", label="Training Accuracy")
plt.plot(train_sizes, val_mean,   "g-o", label="Validation Accuracy")
plt.xlabel("Training Size")
plt.ylabel("Accuracy")
plt.title("Learning Curve - Accuracy (Logistic Regression)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/06_learning_curve_accuracy.png", dpi=150)
plt.close()
print("  → Saved 06_learning_curve_accuracy.png")

# Log-loss learning curve
train_sizes2, train_ll, val_ll = learning_curve(
    LogisticRegression(max_iter=500, random_state=42),
    X_train, y_train, cv=5, scoring="neg_log_loss",
    train_sizes=np.linspace(0.1, 1.0, 10))

plt.figure(figsize=(8, 5))
plt.plot(train_sizes2 / train_sizes2[-1], -train_ll.mean(axis=1), "r-x", label="Training Log-Loss")
plt.plot(train_sizes2 / train_sizes2[-1], -val_ll.mean(axis=1),   "y-x", label="Validation Log-Loss")
plt.xlabel("Training Size (fraction)")
plt.ylabel("Log-Loss")
plt.title("Learning Curve - Log-Loss (Logistic Regression)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/07_learning_curve_logloss.png", dpi=150)
plt.close()
print("  → Saved 07_learning_curve_logloss.png")

# ─────────────────────────────────────────────
# 8. Feature Importance (LR Coefficients)
# ─────────────────────────────────────────────
feature_names = X.columns.tolist()
lr_coef = np.abs(results["Logistic Regression"]["model"].coef_[0])
top_n = 15
top_idx = np.argsort(lr_coef)[-top_n:][::-1]
top_features = [feature_names[i] for i in top_idx]
top_coefs    = lr_coef[top_idx]

plt.figure(figsize=(10, 5))
bars = plt.bar(range(top_n), top_coefs, color="steelblue", edgecolor="black")
plt.xticks(range(top_n), top_features, rotation=45, ha="right", fontsize=9)
plt.ylabel("Absolute Coefficient Value")
plt.title("Feature Coefficients - Logistic Regression (Top 15)")
plt.tight_layout()
plt.savefig(f"{OUT}/08_feature_importance_lr.png", dpi=150)
plt.close()
print("  → Saved 08_feature_importance_lr.png")

# Random Forest Feature Importance
rf_model = results["Random Forest"]["model"]
rf_importance = rf_model.feature_importances_
top_rf_idx = np.argsort(rf_importance)[-top_n:][::-1]
top_rf_features = [feature_names[i] for i in top_rf_idx]
top_rf_imp      = rf_importance[top_rf_idx]

plt.figure(figsize=(10, 5))
plt.bar(range(top_n), top_rf_imp, color="seagreen", edgecolor="black")
plt.xticks(range(top_n), top_rf_features, rotation=45, ha="right", fontsize=9)
plt.ylabel("Feature Importance (Gini)")
plt.title("Feature Importance - Random Forest (Top 15)")
plt.tight_layout()
plt.savefig(f"{OUT}/09_feature_importance_rf.png", dpi=150)
plt.close()
print("  → Saved 09_feature_importance_rf.png")

# ─────────────────────────────────────────────
# 9. Scatter Plot - Feature Importance (LR)
# ─────────────────────────────────────────────
plt.figure(figsize=(10, 5))
plt.scatter(range(top_n), top_coefs, color="purple", s=80, zorder=5)
plt.xticks(range(top_n), top_features, rotation=45, ha="right", fontsize=9)
plt.ylabel("Coefficient Value")
plt.title("Feature Importance - Scatter Plot (Logistic Regression)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/10_scatter_feature_importance.png", dpi=150)
plt.close()
print("  → Saved 10_scatter_feature_importance.png")

# ─────────────────────────────────────────────
# 10. Summary Table
# ─────────────────────────────────────────────
summary = pd.DataFrame({
    "Model":     list(results.keys()),
    "Accuracy":  [f"{r['acc']:.4f}"  for r in results.values()],
    "Precision": [f"{r['prec']:.4f}" for r in results.values()],
    "Recall":    [f"{r['rec']:.4f}"  for r in results.values()],
    "F1-Score":  [f"{r['f1']:.4f}"   for r in results.values()],
    "ROC-AUC":   [f"{r['auc']:.4f}"  for r in results.values()],
    "Log-Loss":  [f"{r['ll']:.4f}"   for r in results.values()],
})

print("\n" + "=" * 60)
print("FINAL SUMMARY TABLE")
print("=" * 60)
print(summary.to_string(index=False))

# Save summary to CSV
summary.to_csv(f"{OUT}/model_summary.csv", index=False)
print("\n  → Saved model_summary.csv")

# ─────────────────────────────────────────────
# 11. Visual Summary Table Plot
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 2.5))
ax.axis("off")
tbl = ax.table(cellText=summary.values, colLabels=summary.columns,
               cellLoc="center", loc="center",
               colColours=["#4472C4"] * len(summary.columns))
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1.2, 1.8)
for (row, col), cell in tbl.get_celld().items():
    if row == 0:
        cell.set_text_props(color="white", fontweight="bold")
plt.title("Model Performance Summary", fontweight="bold", fontsize=13, pad=10)
plt.tight_layout()
plt.savefig(f"{OUT}/11_model_summary_table.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → Saved 11_model_summary_table.png")

print("\n✅ All done! All outputs saved to:", OUT)
