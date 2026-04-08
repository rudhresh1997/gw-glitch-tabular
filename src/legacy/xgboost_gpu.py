import pandas as pd
import numpy as np
import joblib
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import classification_report, f1_score, roc_curve, auc, precision_recall_curve, confusion_matrix
import optuna
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import xgboost as xgb
import shap

# ---------------------------
# 1. Load Data for Tuning (Sampled Data)
# ---------------------------
data_tune = pd.read_csv("sampled_data_cleaned.csv")
X_tune = data_tune.drop(columns=["ml_label"])
y_tune = data_tune["ml_label"]

le = LabelEncoder()
y_tune_encoded = le.fit_transform(y_tune)

unique_classes, class_counts = np.unique(y_tune_encoded, return_counts=True)
class_weights = {cls: len(y_tune_encoded) / count for cls, count in zip(unique_classes, class_counts)}

# ---------------------------
# 2. Stratified Splitting for Tuning
# ---------------------------
X_train_tune, X_temp_tune, y_train_tune, y_temp_tune = train_test_split(
    X_tune, y_tune_encoded, test_size=0.36, stratify=y_tune_encoded, random_state=42
)
X_val_tune, X_test_tune, y_val_tune, y_test_tune = train_test_split(
    X_temp_tune, y_temp_tune, test_size=20/36, stratify=y_temp_tune, random_state=42
)

print("For Tuning:")
print(f"Train shape: {X_train_tune.shape}")
print(f"Validation shape: {X_val_tune.shape}")
print(f"Test shape: {X_test_tune.shape}")

# ---------------------------
# 3. Optuna Tuning on Sampled Data
# ---------------------------
def objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 50, 300)
    max_depth = trial.suggest_int("max_depth", 3, 15)
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.3, 1.0)
    
    pipeline = ImbPipeline([
        ("clf", XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            colsample_bytree=colsample_bytree,
            tree_method="hist",
            device="cuda",
            random_state=42,
            eval_metric="mlogloss",
            objective="multi:softprob",
            num_class=len(np.unique(y_train_tune)),
            n_jobs=1
        ))
    ])
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X_train_tune, y_train_tune, cv=cv, scoring="f1_weighted", n_jobs=-1)
    return np.mean(scores)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100, n_jobs=1, timeout=3600)

print("Best hyperparameters found (from sampled data):")
print(study.best_trial.params)

best_params = study.best_trial.params

# ---------------------------
# 4. Load Full Data for Final Training
# ---------------------------
data_full = pd.read_csv("original_data_cleaned.csv")
X_full = data_full.drop(columns=["ml_label"])
y_full = data_full["ml_label"]
y_full_encoded = le.transform(y_full)  # Reuse the label encoder

# Final splits on full data
X_train, X_temp, y_train, y_temp = train_test_split(
    X_full, y_full_encoded, test_size=0.36, stratify=y_full_encoded, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=20/36, stratify=y_temp, random_state=42
)

print("For Final Training:")
print(f"Train shape: {X_train.shape}")
print(f"Validation shape: {X_val.shape}")
print(f"Test shape: {X_test.shape}")

# ---------------------------
# 5. Final Training with Best Hyperparameters
# ---------------------------
f1_scores = []
seeds = [42 + i for i in range(15)]

start_train_time = time.time()
pipeline = ImbPipeline([
    ("clf", XGBClassifier(
        **best_params,
        tree_method="hist",
        device="cuda",
        random_state=42,
        eval_metric=["mlogloss", "merror"],
        objective="multi:softprob",
        num_class=len(np.unique(y_train)),
        n_jobs=1
    ))
])

sample_weights = np.array([class_weights[cls] for cls in y_train])

history = pipeline.fit(
    X_train, y_train, clf__sample_weight=sample_weights,
    clf__eval_set=[(X_train, y_train), (X_val, y_val)],
    clf__verbose=True
)
train_time = time.time() - start_train_time

start_infer_time = time.time()
y_test_pred_prob = pipeline.predict_proba(X_test)
y_test_pred = np.argmax(y_test_pred_prob, axis=1)
infer_time = time.time() - start_infer_time

for seed in seeds:
    pipeline.named_steps["clf"].random_state = seed
    pipeline.fit(X_train, y_train, clf__sample_weight=sample_weights)
    y_test_pred = pipeline.predict(X_test)
    score = f1_score(y_test, y_test_pred, average="weighted")
    f1_scores.append(score)

print("Classification Report (final data):")
print(classification_report(y_test, y_test_pred))
avg_f1 = np.mean(f1_scores)
std_f1 = np.std(f1_scores)
print(f"\nAverage Weighted F1 over 15 seeds: {avg_f1:.4f} ± {std_f1:.4f}")

score = f1_score(y_test, y_test_pred, average="weighted")
conf_mat = confusion_matrix(y_test, y_test_pred)

metrics_df = pd.DataFrame({
    "Class": unique_classes,
    "F1 Score": f1_score(y_test, y_test_pred, average=None),
})
metrics_df.loc[len(metrics_df)] = ["Average Weighted", score]
metrics_df.loc[len(metrics_df)] = ["Inference Time", infer_time]
metrics_df.loc[len(metrics_df)] = ["Training Time", train_time]
metrics_df.to_csv("./img_xgboost/metrics.csv", index=False)

# ---------------------------
# 6. Plotting (same as before)
# ---------------------------
plt.figure()
plt.plot(pipeline.named_steps["clf"].evals_result()["validation_0"]["mlogloss"], label="Training Loss")
plt.plot(pipeline.named_steps["clf"].evals_result()["validation_1"]["mlogloss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig("./img_xgboost/loss_curve.png")

plt.figure()
plt.plot(pipeline.named_steps["clf"].evals_result()["validation_0"]["merror"], label="Training Error")
plt.plot(pipeline.named_steps["clf"].evals_result()["validation_1"]["merror"], label="Validation Error")
plt.xlabel("Epochs")
plt.ylabel("Classification Error")
plt.legend()
plt.savefig("./img_xgboost/error_curve.png")

plt.figure()
train_accuracy = 1 - np.array(pipeline.named_steps["clf"].evals_result()["validation_0"]["merror"])
val_accuracy = 1 - np.array(pipeline.named_steps["clf"].evals_result()["validation_1"]["merror"])
plt.plot(train_accuracy, label="Training Accuracy")
plt.plot(val_accuracy, label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Training vs Validation Accuracy")
plt.savefig("./img_xgboost/accuracy_curve.png")
plt.show()

plt.figure()
for i in range(len(unique_classes)):
    fpr, tpr, _ = roc_curve(y_test == i, y_test_pred_prob[:, i])
    plt.plot(fpr, tpr, label=f"Class {i} (AUC = {auc(fpr, tpr):.2f})")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.savefig("./img_xgboost/roc_curve.png")

plt.figure()
for i in range(len(unique_classes)):
    precision, recall, _ = precision_recall_curve(y_test == i, y_test_pred_prob[:, i])
    plt.plot(recall, precision, label=f"Class {i}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.savefig("./img_xgboost/pr_curve.png")

plt.figure()
sns.heatmap(conf_mat, annot=False, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("./img_xgboost/conf_matrix.png")

# ---------------------------
# 7. SHAP Analysis and Storage: Multiclass XGBoost (Predicted Class Only)
# ---------------------------

import shap
import numpy as np
import pandas as pd

print("Computing SHAP values for predicted class only using TreeExplainer...")

# Get trained model from pipeline
model = pipeline.named_steps["clf"]

# Predict class probabilities and labels
y_test_pred_prob = model.predict_proba(X_test)
y_test_pred_class = np.argmax(y_test_pred_prob, axis=1)

# Create SHAP TreeExplainer (model-aware)
explainer = shap.TreeExplainer(model)

# Compute SHAP values → returns shape (n_samples, n_features, n_classes)
shap_values_all = explainer.shap_values(X_test)

# Extract SHAP values for the predicted class only
shap_values_pred = shap_values_all[np.arange(len(y_test_pred_class)), :, y_test_pred_class]

# Confirm shape
n_samples, n_features = shap_values_pred.shape
print(f"Extracted SHAP values shape: {shap_values_pred.shape}")

# Format into long-form DataFrame
df_shap_xgb = pd.DataFrame({
    "sample_id": np.repeat(np.arange(n_samples), n_features),
    "feature": list(X_test.columns) * n_samples,
    "shap_value": shap_values_pred.flatten(),
    "feature_value": X_test.values.flatten(),
    "model_name": "XGBoost"
})

# Save to CSV for comparison across models
df_shap_xgb.to_csv("./img_xgboost/shap_values_xgboost.csv", index=False)
print("SHAP values saved to './img_xgboost/shap_values_xgboost.csv'")





# After you compute y_test_pred_class and shap_values_all:

# Map class indices to original string labels if you want names
# (Assuming `le` is your LabelEncoder fit on y_tune)
pred_class_names = le.inverse_transform(y_test_pred_class)

# Build long-form with predicted class
rows = []
features = list(X_test.columns)
for i in range(len(y_test_pred_class)):
    cls_idx = y_test_pred_class[i]
    cls_name = pred_class_names[i]
    shap_row = shap_values_all[i, :, cls_idx]  # (n_features,)
    for f_idx, feat in enumerate(features):
        rows.append({
            "sample_id": i,
            "pred_class_idx": int(cls_idx),
            "pred_class_name": str(cls_name),
            "feature": feat,
            "shap_value": float(shap_row[f_idx]),
            "feature_value": float(X_test.iloc[i, f_idx]),
            "model_name": "XGBoost",
        })

df_shap_xgb = pd.DataFrame(rows)
df_shap_xgb.to_csv("./img_xgboost/shap_values_xgboost_with_class.csv", index=False)

# Create comparison-ready aggregates:
treesap_global = (df_shap_xgb.assign(abs_shap=df_shap_xgb["shap_value"].abs())
                  .groupby("feature", as_index=False)["abs_shap"].mean()
                  .rename(columns={"abs_shap":"mean_abs_shap"}))
treesap_global.to_csv("./img_xgboost/treesap_global.csv", index=False)

treesap_per_class = (df_shap_xgb.assign(abs_shap=df_shap_xgb["shap_value"].abs())
                     .groupby(["pred_class_name","feature"], as_index=False)["abs_shap"].mean()
                     .rename(columns={"pred_class_name":"class","abs_shap":"mean_abs_shap"}))
treesap_per_class.to_csv("./img_xgboost/treesap_per_class.csv", index=False)
