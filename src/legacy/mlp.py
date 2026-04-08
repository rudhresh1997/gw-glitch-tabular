import os, time, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve
)
import optuna
from joblib import Parallel, delayed
import joblib

os.makedirs("./img_mlp", exist_ok=True)


# ------------------- Load Data for Optuna -------------------
df_sampled = pd.read_csv("sampled_data_cleaned.csv")
df_full = pd.read_csv("original_data_cleaned.csv")

target_col = ["ml_label"]
num_features = [c for c in df_full.columns if df_full[c].dtype in ["int64", "float64"] and c not in target_col]
feature_names = num_features

X_sampled = df_sampled.drop(columns=["ml_label"]).to_numpy(dtype=np.float32)
y_sampled = df_sampled["ml_label"].astype(str).to_numpy()

X_full = df_full.drop(columns=["ml_label"]).to_numpy(dtype=np.float32)
y_full = df_full["ml_label"].astype(str).to_numpy()

# Label encoder (shared for both)
le = LabelEncoder()
y_sampled_enc = le.fit_transform(y_sampled)
y_full_enc = le.transform(y_full)
unique_classes = le.classes_
n_classes = len(unique_classes)
n_features = X_full.shape[1]
joblib.dump(le, "label_encoder.pkl")


# ------------------- Define MLP -------------------
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation="ReLU", dropout=0.2):
        super().__init__()
        act = getattr(nn, activation)
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev_dim, h), act(), nn.Dropout(dropout)]
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ------------------- Optuna on Sampled -------------------
def objective(trial):
    hidden = trial.suggest_categorical("layers", [[64, 32], [128, 64, 32], [256, 128, 64], [512, 256, 128]])
    activation = trial.suggest_categorical("activation", ["ReLU", "LeakyReLU", "GELU", "Mish"])
    dropout = trial.suggest_float("dropout", 0.0, 0.4)
    lr = trial.suggest_loguniform("learning_rate", 1e-4, 1e-2)
    batch_size = 1024

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    f1_scores = []

    for train_idx, val_idx in skf.split(X_sampled, y_sampled_enc):
        X_tr, X_va = X_sampled[train_idx], X_sampled[val_idx]
        y_tr, y_va = y_sampled_enc[train_idx], y_sampled_enc[val_idx]

        scaler = StandardScaler().fit(X_tr)
        X_tr, X_va = scaler.transform(X_tr), scaler.transform(X_va)

        train_ds = TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                                 torch.tensor(y_tr, dtype=torch.long))
        val_ds = TensorDataset(torch.tensor(X_va, dtype=torch.float32),
                               torch.tensor(y_va, dtype=torch.long))

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SimpleMLP(n_features, hidden, n_classes, activation, dropout).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr)

        for epoch in range(5):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()

        # Validation
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds.append(model(xb).argmax(1).cpu().numpy())
                trues.append(yb.cpu().numpy())
        preds = np.concatenate(preds); trues = np.concatenate(trues)
        f1_scores.append(f1_score(trues, preds, average="macro"))

    return np.mean(f1_scores)


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)
best_params = study.best_params
print("Best hyperparams:", best_params)


# ------------------- Final Training on Full Data -------------------
def train_single_seed(seed, params):
    hidden = params["layers"]
    activation = params["activation"]
    dropout = params["dropout"]
    lr = params["learning_rate"]

    train_idx, test_idx = train_test_split(
        np.arange(len(y_full_enc)), test_size=0.36, stratify=y_full_enc, random_state=seed
    )
    val_idx, test_idx = train_test_split(
        test_idx, test_size=20/36, stratify=y_full_enc[test_idx], random_state=seed
    )

    X_train, X_val, X_test = X_full[train_idx], X_full[val_idx], X_full[test_idx]
    y_train, y_val, y_test = y_full_enc[train_idx], y_full_enc[val_idx], y_full_enc[test_idx]

    scaler = StandardScaler().fit(X_train)
    X_train, X_val, X_test = scaler.transform(X_train), scaler.transform(X_val), scaler.transform(X_test)

    # 🔧 Return these for Captum later
    X_test_final = X_test
    y_test_final = y_test

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                             torch.tensor(y_train, dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                           torch.tensor(y_val, dtype=torch.long))
    test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                            torch.tensor(y_test, dtype=torch.long))

    train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1024)
    test_loader = DataLoader(test_ds, batch_size=1024)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SimpleMLP(n_features, hidden, n_classes, activation, dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # Training
    start_train_time = time.time()
    for epoch in range(20):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
    train_time = time.time() - start_train_time

    # Inference
    model.eval()
    preds, probs, trues = [], [], []
    start_infer_time = time.time()
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            preds.append(logits.argmax(1).cpu().numpy())
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
            trues.append(yb.cpu().numpy())
    infer_time = time.time() - start_infer_time

    preds = np.concatenate(preds); probs = np.concatenate(probs); trues = np.concatenate(trues)
    f1 = f1_score(trues, preds, average="weighted")

    # 🔧 Return X_test_final and y_test_final
    return (f1, preds, probs, model, train_time, infer_time, trues, X_test_final, y_test_final)


# Run 15 seeds
seeds = list(range(42, 57))
results = Parallel(n_jobs=1)(delayed(train_single_seed)(s, best_params) for s in seeds)

# Aggregate
f1_scores = [r[0] for r in results]
train_times = [r[4] for r in results]
infer_times = [r[5] for r in results]

avg_f1, std_f1 = np.mean(f1_scores), np.std(f1_scores)
avg_train_time, avg_infer_time = np.mean(train_times), np.mean(infer_times)

print(f"\nFinal Average Weighted F1: {avg_f1:.4f} ± {std_f1:.4f}")
print(f"Average Training Time: {avg_train_time:.4f} seconds")
print(f"Average Inference Time: {avg_infer_time:.4f} seconds")

# Best model
best_index = np.argmax(f1_scores)
(best_score, y_test_pred, y_test_pred_prob, best_model,
 train_time, infer_time, y_test, X_test_final, y_test_final) = results[best_index]

print("\nClassification Report (Best Seed):")
print(classification_report(y_test, y_test_pred, target_names=unique_classes))

conf_mat = confusion_matrix(y_test, y_test_pred)

# Per-class F1
f1_per_class = f1_score(y_test, y_test_pred, average=None, labels=np.arange(n_classes), zero_division=0)

metrics_df = pd.DataFrame({"Class": unique_classes, "F1 Score": f1_per_class})
metrics_df.loc[len(metrics_df)] = ["Average Weighted", best_score]
metrics_df.loc[len(metrics_df)] = ["Inference Time", infer_time]
metrics_df.loc[len(metrics_df)] = ["Training Time", train_time]
metrics_df.to_csv("./img_mlp/metrics.csv", index=False)


# ------------------- Plots -------------------
plt.figure()
for i, cls in enumerate(unique_classes):
    fpr, tpr, _ = roc_curve(y_test == i, y_test_pred_prob[:, i])
    plt.plot(fpr, tpr, label=f"{cls} (AUC={auc(fpr, tpr):.2f})")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.legend(); plt.savefig("./img_mlp/roc_curve.png")

plt.figure()
for i, cls in enumerate(unique_classes):
    precision, recall, _ = precision_recall_curve(y_test == i, y_test_pred_prob[:, i])
    plt.plot(recall, precision, label=f"{cls}")
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.legend()
plt.savefig("./img_mlp/pr_curve.png")

plt.figure()
sns.heatmap(conf_mat, annot=False, fmt="d", cmap="Blues",
            xticklabels=unique_classes, yticklabels=unique_classes)
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.savefig("./img_mlp/conf_matrix.png")


# === Captum Attribution with Safe Batching for Best MLP Model ===
from captum.attr import IntegratedGradients, DeepLift
from scipy.stats import spearmanr

os.makedirs("./img_mlp", exist_ok=True)

ig = IntegratedGradients(best_model)
dl = DeepLift(best_model)

device = next(best_model.parameters()).device
baseline = torch.zeros((1, X_test_final.shape[1]), dtype=torch.float32).to(device)

def batched_attr(inputs_t, targets_idx, batch_size=128, n_steps=50, explainer="ig"):
    N = inputs_t.shape[0]
    out = np.zeros((N, inputs_t.shape[1]), dtype=np.float32)
    for s in range(0, N, batch_size):
        e = min(s + batch_size, N)
        x = inputs_t[s:e]
        t = targets_idx[s:e]
        if explainer == "ig":
            a = ig.attribute(
                x, target=t,
                baselines=baseline.repeat(e-s, 1),
                n_steps=n_steps,
                internal_batch_size=batch_size
            )
        else:
            a = dl.attribute(
                x, target=t,
                baselines=baseline.repeat(e-s, 1)
            )
        out[s:e] = a.detach().cpu().abs().numpy()
        torch.cuda.empty_cache()
    return out

inputs_t = torch.tensor(X_test_final, dtype=torch.float32).to(device)
preds_idx = torch.tensor(y_test_pred, dtype=torch.long).to(device)

abs_attrs_ig = batched_attr(inputs_t, preds_idx, batch_size=128, n_steps=50, explainer="ig")

df_attrs = pd.DataFrame(abs_attrs_ig, columns=feature_names)

# Per-class
per_class_rows = []
for k, cname in enumerate(unique_classes):
    mask = (y_test_pred == k)
    if mask.any():
        mean_abs = df_attrs[mask].mean(axis=0)
        std_abs = df_attrs[mask].std(axis=0, ddof=1).fillna(0.0)
    else:
        mean_abs = pd.Series(0.0, index=feature_names)
        std_abs = pd.Series(0.0, index=feature_names)
    for feat in feature_names:
        per_class_rows.append({
            "class": cname,
            "feature": feat,
            "mean_abs_attr": float(mean_abs[feat]),
            "std_abs_attr": float(std_abs[feat]),
        })

per_class_df = pd.DataFrame(per_class_rows)
per_class_df["rank_within_class"] = per_class_df.groupby("class")["mean_abs_attr"]\
                                               .rank(ascending=False, method="dense")

# Global
global_df = df_attrs.mean(axis=0).reset_index()
global_df.columns = ["feature", "mean_abs_attr"]
global_df["rank_global"] = global_df["mean_abs_attr"].rank(ascending=False, method="dense")

per_class_path = "./img_mlp/captum_ig_per_class.csv"
global_path = "./img_mlp/captum_ig_global.csv"
per_class_df.to_csv(per_class_path, index=False)
global_df.to_csv(global_path, index=False)
print(f"Saved: {per_class_path}, {global_path}")


# === Spearman Correlation: Captum IG (MLP) vs XGBoost TreeSHAP ===
import pandas as pd
from scipy.stats import spearmanr

xgb_global_csv = "./img_xgboost/treesap_global.csv"
xgb_per_class_csv = "./img_xgboost/treesap_per_class.csv"

try:
    # Global correlation
    xgb_g = pd.read_csv(xgb_global_csv).set_index("feature")
    mlp_g = pd.read_csv("./img_mlp/captum_ig_global.csv").set_index("feature")
    common_feats = xgb_g.index.intersection(mlp_g.index)

    rho_g, p_g = spearmanr(
        xgb_g.loc[common_feats, "mean_abs_shap"].rank(ascending=False),
        mlp_g.loc[common_feats, "mean_abs_attr"].rank(ascending=False),
    )
    print(f"[GLOBAL] Spearman ρ(TreeSHAP vs Best MLP Captum IG) = {rho_g:.3f} (p={p_g:.3g})")

    # Per-class correlation
    xgb_c = pd.read_csv(xgb_per_class_csv)
    mlp_c = pd.read_csv("./img_mlp/captum_ig_per_class.csv")
    classes_to_check = sorted(set(xgb_c["class"]).intersection(set(mlp_c["class"])))

    for cname in classes_to_check:
        xx = xgb_c[xgb_c["class"] == cname].set_index("feature")
        mm = mlp_c[mlp_c["class"] == cname].set_index("feature")
        feats = xx.index.intersection(mm.index)
        if len(feats) < 2:
            continue
        rho, p = spearmanr(
            xx.loc[feats, "mean_abs_shap"].rank(ascending=False),
            mm.loc[feats, "mean_abs_attr"].rank(ascending=False),
        )
        print(f"[CLASS {cname}] Spearman ρ = {rho:.3f} (p={p:.3g})")

except FileNotFoundError as e:
    print(f"Spearman correlation skipped: {e}")


# === Export Correlation Results to CSV ===
corr_rows = []

# Add global correlation
try:
    corr_rows.append({
        "class": "GLOBAL",
        "spearman_rho": rho_g,
        "p_value": p_g
    })
except Exception as e:
    print("[WARN] Global correlation not available:", e)

# Add per-class correlations
try:
    for cname in classes_to_check:
        xx = xgb_c[xgb_c["class"] == cname].set_index("feature")
        mm = mlp_c[mlp_c["class"] == cname].set_index("feature")
        feats = xx.index.intersection(mm.index)
        if len(feats) < 2:
            corr_rows.append({
                "class": cname,
                "spearman_rho": np.nan,
                "p_value": np.nan
            })
            continue
        rho, p = spearmanr(
            xx.loc[feats, "mean_abs_shap"].rank(ascending=False),
            mm.loc[feats, "mean_abs_attr"].rank(ascending=False),
        )
        corr_rows.append({
            "class": cname,
            "spearman_rho": rho,
            "p_value": p
        })
except Exception as e:
    print("[WARN] Per-class correlation export skipped:", e)

# Save all correlations to CSV
corr_df = pd.DataFrame(corr_rows)
corr_df.to_csv("./img_mlp/captum_vs_treeshap_corr.csv", index=False)
print("Saved correlations CSV: ./img_mlp/captum_vs_treeshap_corr.csv")

