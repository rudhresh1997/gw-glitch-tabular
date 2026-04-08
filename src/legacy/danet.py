import optuna
from pytorch_tabular.utils.nn_utils import OOMException
from pytorch_tabular import TabularModel
from pytorch_tabular.models import DANetConfig
from pytorch_tabular.config import DataConfig, TrainerConfig, OptimizerConfig
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import torch
import pandas as pd


# Load Data
df = pd.read_csv("sampled_data_cleaned.csv")  # Modify with actual path

# Splitting dataset
train_df, test_df = train_test_split(df, test_size=0.36, random_state=42, stratify=df["ml_label"])
val_df, test_df = train_test_split(test_df, test_size=20/36, random_state=42, stratify=test_df["ml_label"])

target_col = ["ml_label"]
num_features = [col for col in df.columns if df[col].dtype in ["int64", "float64"] and col != target_col]

# Override the function to load the full checkpoint properly
def custom_load_best_model(self):
    """Custom function to load the best model without the 'weights_only' restriction."""
    if self.trainer.checkpoint_callback is not None and self.trainer.checkpoint_callback.best_model_path != "":
        ckpt_path = self.trainer.checkpoint_callback.best_model_path
        print(f"Loading best model from: {ckpt_path}")
        
        # Use torch.load instead of pl_load, removing 'weights_only' argument
        ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage, weights_only=False)  
        
        # Load model state
        self.model.load_state_dict(ckpt["state_dict"])
    else:
        print("No best model found!")

# Patch the function into TabularModel
TabularModel.load_best_model = custom_load_best_model

# Define Optuna Objective Function
def objective(trial):
    n_layers = trial.suggest_categorical("n_layers", [4, 8])
    abstlay_dim_1 = 32
    abstlay_dim_2 = 64
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.25)
    learning_rate = trial.suggest_float("learning_rate", 5e-4, 3e-3, log=True)

    # Cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(df, df[target_col])):
        train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]

        data_config = DataConfig(
            target=target_col,
            continuous_cols=num_features,
            categorical_cols=[],
            num_workers=0,
            normalize_continuous_features=True
        )

        model_config = DANetConfig(
            task="classification",
            n_layers = n_layers,
            abstlay_dim_1 = abstlay_dim_1,
            abstlay_dim_2 = abstlay_dim_2,
            dropout_rate = dropout_rate,
            learning_rate = learning_rate
        )

        trainer_config = TrainerConfig(
            batch_size=512,
            max_epochs=8,
            early_stopping_patience=2,
            accelerator="gpu",
            devices=1
        )

        optimizer_config = OptimizerConfig(optimizer='AdamW')

        tabular_model = TabularModel(
            data_config=data_config,
            model_config=model_config,
            trainer_config=trainer_config,
            optimizer_config=optimizer_config
        )

        try:
            #OOM SAFE BLOCK
            tabular_model.fit(train=train_df, validation=val_df)
        except OOMException:
            print(f"Trial pruned at fold {fold_idx} due to OOM (PT-Tabular).")
            raise optuna.exceptions.TrialPruned()
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"Trial pruned at fold {fold_idx} due to CUDA OOM.")
                raise optuna.exceptions.TrialPruned()
            else:
                raise

        tabular_model.fit(train=train_df, validation=val_df)
        pred_df = tabular_model.predict(val_df)
        y_true = val_df[target_col].values
        y_pred = pred_df["ml_label_prediction"].values

        f1 = f1_score(y_true, y_pred, average="macro")
        f1_scores.append(f1)

        trial.report(f1, step=fold_idx)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return sum(f1_scores) / len(f1_scores)  # Average F1-score over folds

# Run Optuna Optimization
study = optuna.create_study(
    direction="maximize",
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=2)
)

study.optimize(objective, n_trials=100)

print("Best hyperparameters found:")
print(study.best_trial.params)
print(f"Best cross-validated weighted F1: {study.best_value:.4f}")

best_params = study.best_trial.params

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
)
from pytorch_tabular import TabularModel
from pytorch_tabular.models import DANetConfig
from pytorch_tabular.config import DataConfig, TrainerConfig, OptimizerConfig
from sklearn.model_selection import train_test_split

# Load Data
df = pd.read_csv("original_data_cleaned.csv")  # Modify with actual path

# Splitting dataset
train_df, test_df = train_test_split(df, test_size=0.36, random_state=42, stratify=df["ml_label"])
val_df, test_df = train_test_split(test_df, test_size=20/36, random_state=42, stratify=test_df["ml_label"])

target_col = ["ml_label"]
num_features = [col for col in df.columns if df[col].dtype in ["int64", "float64"] and col not in target_col]
unique_classes = df["ml_label"].unique()

# Define Training Function
def train_single_seed(seed, best_params):
    
    data_config = DataConfig(
        target=target_col,
        continuous_cols=num_features,
        categorical_cols=[],
        num_workers=0,
        normalize_continuous_features=True
    )

    model_config = DANetConfig(
        task="classification",
        **best_params
    )

    trainer_config = TrainerConfig(
        max_epochs=50,
        seed=seed,
        batch_size=1024,
        early_stopping_patience=5,
        accelerator="gpu",
        devices=1
    )

    optimizer_config = OptimizerConfig(optimizer='AdamW')

    # Initialize model
    tabular_model = TabularModel(
        data_config=data_config,
        model_config=model_config,
        trainer_config=trainer_config,
        optimizer_config=optimizer_config
    )

    # Measure Training Time
    start_train_time = time.time()
    tabular_model.fit(train=train_df, validation=val_df)
    train_time = time.time() - start_train_time

    # Measure Inference Time on Test Set
    X_test_sample = test_df.sample(n=10112, random_state=42)
    y_test = X_test_sample[target_col].values

    start_infer_time = time.time()
    pred_df = tabular_model.predict(X_test_sample)
    infer_time = time.time() - start_infer_time

    y_test_pred = pred_df["ml_label_prediction"].values
    y_test_pred_prob = pred_df[[col for col in pred_df.columns if "_proba" in col]].values

    f1 = f1_score(y_test, y_test_pred, average="weighted")
    print(f"Seed {seed} - Weighted F1 Score: {f1:.4f}")
    
    return (f1, y_test_pred, y_test_pred_prob, tabular_model, train_time, infer_time)

# Run Training for Multiple Seeds
seeds = [42 + i for i in range(15)]
results = Parallel(n_jobs=1)(delayed(train_single_seed)(seed, best_params) for seed in seeds)

# Aggregate Results
f1_scores = [res[0] for res in results]
train_times = [res[4] for res in results]
infer_times = [res[5] for res in results]

avg_f1 = np.mean(f1_scores)
std_f1 = np.std(f1_scores)
avg_train_time = np.mean(train_times)
avg_infer_time = np.mean(infer_times)

print(f"\nFinal Average Weighted F1: {avg_f1:.4f} ± {std_f1:.4f}")
print(f"Average Training Time: {avg_train_time:.4f} seconds")
print(f"Average Inference Time for 100 samples: {avg_infer_time:.4f} seconds")

# Select Best Seed Model
best_index = np.argmax(f1_scores)
best_score, y_test_pred, y_test_pred_prob, best_model, train_time, infer_time = results[best_index]

X_test_sample = test_df.sample(n=10112, random_state=42)
y_test = X_test_sample[target_col].values

print("\nClassification Report (Best Seed):")
print(classification_report(y_test, y_test_pred))

conf_mat = confusion_matrix(y_test, y_test_pred)
print(f"\nConfusion Matrix:\n{conf_mat}")

# Compute F1 scores properly across all classes
f1_scores = f1_score(
    y_test, 
    y_test_pred, 
    average=None, 
    labels=unique_classes, 
    zero_division=0  # Handle missing classes gracefully
)

# Save Metrics
metrics_df = pd.DataFrame({
    "Class": unique_classes,
    "F1 Score": f1_scores,
})
metrics_df.loc[len(metrics_df)] = ["Average Weighted", best_score]
metrics_df.loc[len(metrics_df)] = ["Inference Time", infer_time]
metrics_df.loc[len(metrics_df)] = ["Training Time", train_time]
metrics_df.to_csv("./img_danet/metrics.csv", index=False)


# Plot ROC Curve
plt.figure()
for i, cls in enumerate(unique_classes):
    fpr, tpr, _ = roc_curve(y_test == cls, y_test_pred_prob[:, i])
    plt.plot(fpr, tpr, label=f"Class {cls} (AUC = {auc(fpr, tpr):.2f})")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.savefig("./img_danet/roc_curve.png")

# Plot Precision-Recall Curve
plt.figure()
for i, cls in enumerate(unique_classes):
    precision, recall, _ = precision_recall_curve(y_test == cls, y_test_pred_prob[:, i])
    plt.plot(recall, precision, label=f"Class {cls}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.savefig("./img_danet/pr_curve.png")

# Plot Confusion Matrix
plt.figure()
sns.heatmap(conf_mat, annot=False, fmt="d", cmap="Blues", xticklabels=unique_classes, yticklabels=unique_classes)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("./img_danet/conf_matrix.png")






# === Export clean DANet inference model, run Captum, save CSVs, and compare to TreeSHAP ===
import os, json, numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from captum.attr import IntegratedGradients, DeepLift
from scipy.stats import spearmanr
from typing import Optional

os.makedirs("./img_danet", exist_ok=True)

# ---------- 1) Export a clean inference model with frozen normalization ----------
scaler = StandardScaler().fit(train_df[num_features].to_numpy(dtype=np.float32))

class _LightningWrapper(nn.Module):
    """
    Wrapper for DANet LightningModule.
    Prepares dict input {continuous, categorical} as DANet expects.
    """
    def __init__(self, lightning_module, mu: np.ndarray, sigma: np.ndarray, device="cpu"):
        super().__init__()
        self.lm = lightning_module.eval()
        for p in self.lm.parameters():
            p.requires_grad_(False)
        self.register_buffer("mu", torch.tensor(mu.astype(np.float32)))
        self.register_buffer("sigma", torch.tensor(sigma.astype(np.float32)))
        self.device = torch.device(device)
        self.to(self.device)

    def forward(self, X):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
        if not torch.is_tensor(X):
            X = torch.tensor(X, dtype=torch.float32)
        X = X.to(self.device).float()
        Xn = (X - self.mu) / self.sigma

        # >>> MAKE CONTIGUOUS <<<
        Xn = Xn.contiguous()

        N = Xn.shape[0]
        batch = {
            "continuous": Xn,
            "categorical": torch.empty((N, 0), dtype=torch.long, device=self.device),
        }
        out = self.lm(batch)
        if isinstance(out, dict):
            if "logits" in out: return out["logits"]
            if "output" in out: return out["output"]
        if isinstance(out, (list, tuple)) and len(out) > 0:
            return out[0]
        return out  # assume tensor

# --- CHANGE: Force LightningWrapper for DANet ---
def build_exported_model(tabular_model, scaler, device="cpu"):
    lm = tabular_model.model
    mu = scaler.mean_
    sigma = scaler.scale_
    print("[ExportedDANet] Using LightningWrapper with dict input")
    return _LightningWrapper(lm, mu, sigma, device=device).eval()

# Choose device; CUDA recommended if available
device = "cuda" if torch.cuda.is_available() else "cpu"
exported = build_exported_model(best_model, scaler, device=device)

# ---------- 2) Forward once to discover true class count ----------
X_all = test_df[num_features].to_numpy(dtype=np.float32)
X_all_t = torch.from_numpy(X_all).to(device)

with torch.no_grad():
    logits = exported(X_all_t)
    probs = torch.softmax(logits, dim=1)
    y_pred_idx = torch.argmax(probs, dim=1).cpu().numpy()

num_model_classes = logits.shape[1]

# ---------- 3) Resolve class names ----------
class_names = None
try:
    enc = getattr(best_model.datamodule, "_target_transform", None)
    if enc is not None and hasattr(enc, "classes_"):
        cand = list(pd.Index(enc.classes_).astype(str))
        if len(cand) == num_model_classes:
            class_names = cand
            print("[labels] Using datamodule._target_transform.classes_")
except Exception:
    pass

if class_names is None:
    try:
        le = LabelEncoder().fit(train_df["ml_label"].astype(str))
        cand = list(pd.Index(le.classes_).astype(str))
        if len(cand) == num_model_classes:
            class_names = cand
            print("[labels] Using LabelEncoder() fitted on train_df['ml_label']")
    except Exception:
        pass

if class_names is None:
    class_names = [f"class_{i}" for i in range(num_model_classes)]
    print("[labels] Falling back to generic class names (class_0..class_{C-1})")

n_classes = len(class_names)
n_features = len(num_features)

with open("./img_danet/attr_metadata_danet.json", "w") as f:
    json.dump({
        "model": "DANet",
        "device": device,
        "n_features": n_features,
        "features": num_features,
        "n_classes": n_classes,
        "class_names": class_names,
        "normalization": "StandardScaler(train_df)"
    }, f, indent=2)

# ---------- 4) Captum attributions ----------
ig = IntegratedGradients(exported)
dl = DeepLift(exported)  # optional check
baseline = torch.zeros_like(X_all_t)

def batched_attr(inputs_t, targets_idx, batch_size=2048 if device == "cuda" else 512, n_steps=50, explainer="ig"):
    N = inputs_t.shape[0]
    out = np.zeros((N, n_features), dtype=np.float32)
    for s in range(0, N, batch_size):
        e = min(s + batch_size, N)
        x = inputs_t[s:e]
        t = torch.tensor(targets_idx[s:e], device=device)
        if explainer == "ig":
            a = ig.attribute(x, target=t, baselines=baseline[s:e],
                             n_steps=n_steps, internal_batch_size=batch_size)
        else:
            a = dl.attribute(x, target=t, baselines=baseline[s:e])
        out[s:e] = a.detach().cpu().abs().numpy()
    return out

abs_attrs_ig = batched_attr(X_all_t, y_pred_idx, n_steps=50, explainer="ig")

# ---------- 5) Aggregations ----------
df_attrs = pd.DataFrame(abs_attrs_ig, columns=num_features)

per_class_rows = []
for k in range(num_model_classes):
    mask = (y_pred_idx == k)
    if mask.any():
        mean_abs = df_attrs[mask].mean(axis=0)
        std_abs  = df_attrs[mask].std(axis=0, ddof=1).fillna(0.0)
    else:
        mean_abs = pd.Series(0.0, index=num_features)
        std_abs  = pd.Series(0.0, index=num_features)
    cname = class_names[k]
    for feat in num_features:
        per_class_rows.append({
            "class": cname,
            "feature": feat,
            "mean_abs_attr": float(mean_abs[feat]),
            "std_abs_attr":  float(std_abs[feat]),
        })

per_class_df = pd.DataFrame(per_class_rows)
per_class_df["rank_within_class"] = per_class_df.groupby("class")["mean_abs_attr"]\
                                               .rank(ascending=False, method="dense")

global_df = df_attrs.mean(axis=0).reset_index()
global_df.columns = ["feature", "mean_abs_attr"]
global_df["rank_global"] = global_df["mean_abs_attr"].rank(ascending=False, method="dense")

per_class_path = "./img_danet/captum_ig_per_class.csv"
global_path    = "./img_danet/captum_ig_global.csv"
per_class_df.to_csv(per_class_path, index=False)
global_df.to_csv(global_path, index=False)
print(f"Saved: {per_class_path}, {global_path}")

# ---------- 6) Compare to XGBoost TreeSHAP ----------
xgb_global_csv = "./img_xgboost/treesap_global.csv"
xgb_per_class_csv = "./img_xgboost/treesap_per_class.csv"

xgb_g = pd.read_csv(xgb_global_csv).set_index("feature")
danet_g = pd.read_csv(global_path).set_index("feature")
common_feats = xgb_g.index.intersection(danet_g.index)

rho_g, p_g = spearmanr(
    xgb_g.loc[common_feats, "mean_abs_shap"].rank(ascending=False),
    danet_g.loc[common_feats, "mean_abs_attr"].rank(ascending=False),
)
print(f"[GLOBAL] Spearman ρ(TreeSHAP vs Captum IG, DANet) = {rho_g:.3f} (p={p_g:.3g})")

try:
    xgb_c = pd.read_csv(xgb_per_class_csv)
    danet_c = pd.read_csv(per_class_path)
    classes_to_check = sorted(set(xgb_c["class"]).intersection(set(danet_c["class"])))
    for cname in classes_to_check:
        xx = xgb_c[xgb_c["class"] == cname].set_index("feature")
        mm = danet_c[danet_c["class"] == cname].set_index("feature")
        feats = xx.index.intersection(mm.index)
        if len(feats) < 2: continue
        rho, p = spearmanr(
            xx.loc[feats, "mean_abs_shap"].rank(ascending=False),
            mm.loc[feats, "mean_abs_attr"].rank(ascending=False),
        )
        print(f"[CLASS {cname}] Spearman ρ = {rho:.3f} (p={p:.3g})")
except FileNotFoundError:
    print("Per-class comparison skipped: ./img_xgboost/treesap_per_class.csv not found.")




# === Export Correlation Results to CSV (DANet) ===
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
        mm = danet_c[danet_c["class"] == cname].set_index("feature")
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
out_path = "./img_danet/captum_vs_treeshap_corr.csv"
corr_df.to_csv(out_path, index=False)
print(f"Saved correlations CSV: {out_path}")
