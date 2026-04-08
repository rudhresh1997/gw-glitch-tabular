import optuna
from pytorch_tabular.utils.nn_utils import OOMException
from pytorch_tabular import TabularModel
from pytorch_tabular.models import TabTransformerConfig
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
    
    valid_pairs = [
        (16, 2),
        (16, 4),
        (24, 2),
        (24, 4),
        (32, 2),
        (32, 4),
        (32, 8),
        (48, 4),
        (48, 6),
        (48, 8),
    ]

    input_embed_dim, num_heads = trial.suggest_categorical("embed_dim_head_pair", valid_pairs)

    num_attn_blocks = trial.suggest_int("num_attn_blocks", 2, 4)
    attn_dropout = trial.suggest_float("attn_dropout", 0.1, 0.3)
    ff_dropout = trial.suggest_float("ff_dropout", 0.1, 0.3)
    ff_hidden_multiplier = trial.suggest_categorical("ff_hidden_multiplier", [2, 3])
    transformer_activation = trial.suggest_categorical("transformer_activation", ["GEGLU", "ReGLU", "SwiGLU"])
    embedding_dropout = trial.suggest_float("embedding_dropout", 0.0, 0.1)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-2)

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

        model_config = TabTransformerConfig(
            task="classification",
            input_embed_dim=input_embed_dim,
            num_heads=num_heads,
            num_attn_blocks=num_attn_blocks,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            ff_hidden_multiplier=ff_hidden_multiplier,
            transformer_activation=transformer_activation,
            embedding_dropout=embedding_dropout,
            learning_rate=learning_rate
        )

        trainer_config = TrainerConfig(
            batch_size=512,
            max_epochs=20,
            early_stopping_patience=5,
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
            # OOM-safe training block
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

best_embed_dim, best_num_heads = best_params.pop("embed_dim_head_pair")
best_params["input_embed_dim"] = best_embed_dim
best_params["num_heads"] = best_num_heads

import os
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
from pytorch_tabular.models import TabTransformerConfig
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

    model_config = TabTransformerConfig(
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
print(f"Average Inference Time on sampled test subset: {avg_infer_time:.4f} seconds")

# Select Best Seed Model
best_index = np.argmax(f1_scores)
best_score, _, _, best_model, train_time, infer_time = results[best_index]

# --------------------------------------------------
# FULL TEST-SET EVALUATION FOR CONFUSION MATRIX
# --------------------------------------------------
X_test_full = test_df.copy()
y_test_full = X_test_full["ml_label"].astype(str).values

pred_df_full = best_model.predict(X_test_full)
y_test_pred_full = pred_df_full["ml_label_prediction"].astype(str).values
y_test_pred_prob_full = pred_df_full[[col for col in pred_df_full.columns if "_proba" in col]].values

# Resolve exact class order from the trained model
class_labels = None
try:
    enc = getattr(best_model.datamodule, "_target_transform", None)
    if enc is not None and hasattr(enc, "classes_"):
        class_labels = list(pd.Index(enc.classes_).astype(str))
        print("[labels] Using best_model.datamodule._target_transform.classes_")
except Exception:
    pass

if class_labels is None:
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder().fit(train_df["ml_label"].astype(str))
    class_labels = list(pd.Index(le.classes_).astype(str))
    print("[labels] Falling back to LabelEncoder fitted on train_df['ml_label']")

print(f"[labels] Number of classes in encoder order: {len(class_labels)}")
print(f"[labels] Classes: {class_labels}")

print("\nClassification Report (Best Seed, FULL TEST SET):")
print(classification_report(y_test_full, y_test_pred_full, labels=class_labels, zero_division=0))

conf_mat = confusion_matrix(y_test_full, y_test_pred_full, labels=class_labels)
print(f"\nConfusion Matrix shape: {conf_mat.shape}")
print(f"\nConfusion Matrix:\n{conf_mat}")

# Per-class F1 in the correct class order
per_class_f1 = f1_score(
    y_test_full,
    y_test_pred_full,
    average=None,
    labels=class_labels,
    zero_division=0
)

# Save confusion matrix and label order
os.makedirs("./img_tabtransformer", exist_ok=True)

pd.DataFrame(conf_mat, index=class_labels, columns=class_labels).to_csv("./img_tabtransformer/confusion_matrix_full.csv")
pd.DataFrame({"class": class_labels, "f1_score": per_class_f1}).to_csv("./img_tabtransformer/per_class_f1_full.csv", index=False)

# Save metrics
metrics_df = pd.DataFrame({
    "Class": class_labels,
    "F1 Score": per_class_f1,
})
metrics_df.loc[len(metrics_df)] = ["Average Weighted", best_score]
metrics_df.loc[len(metrics_df)] = ["Inference Time", infer_time]
metrics_df.loc[len(metrics_df)] = ["Training Time", train_time]
metrics_df.to_csv("./img_tabtransformer/metrics.csv", index=False)


# Plot ROC Curve
plt.figure()
for i, cls in enumerate(unique_classes):
    fpr, tpr, _ = roc_curve(y_test == cls, y_test_pred_prob[:, i])
    plt.plot(fpr, tpr, label=f"Class {cls} (AUC = {auc(fpr, tpr):.2f})")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.savefig("./img_tabtransformer/roc_curve.png")

# Plot Precision-Recall Curve
plt.figure()
for i, cls in enumerate(unique_classes):
    precision, recall, _ = precision_recall_curve(y_test == cls, y_test_pred_prob[:, i])
    plt.plot(recall, precision, label=f"Class {cls}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.savefig("./img_tabtransformer/pr_curve.png")

# --------------------------------------------------
# PRD-OPTIMIZED CONFUSION MATRIX PLOTS
# --------------------------------------------------
row_sums = conf_mat.sum(axis=1, keepdims=True)
conf_mat_norm = np.divide(conf_mat, row_sums, where=row_sums != 0)

plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 18,
    "axes.titlesize": 20,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10
})

display_map = {
    "Scattered_Light": "Scattered\nLight",
    "Fast_Scattering": "Fast\nScattering",
    "Blip_Low_Frequency": "Blip-LF",
    "Low_Frequency_Burst": "LF-Burst",
    "Extremely_Loud": "Extremely\nLoud",
    "No_Glitch": "No\nGlitch",
    "Repeating_Blips": "Repeating\nBlips",
    "Low_Frequency_Lines": "LF-Lines",
    "Air_Compressor": "Air\nCompressor",
    "Violin_Mode": "Violin\nMode",
    "Power_Line": "Power\nLine",
    "Light_Modulation": "Light\nModulation",
    "None_of_the_Above": "None of\nthe Above"
}
display_labels = [display_map.get(x, x) for x in class_labels]

# Row-normalized version for main paper
fig, ax = plt.subplots(figsize=(13, 11))
hm = sns.heatmap(
    conf_mat_norm,
    cmap="Blues",
    vmin=0.0,
    vmax=1.0,
    square=True,
    xticklabels=display_labels,
    yticklabels=display_labels,
    linewidths=0.05,
    linecolor="lightgray",
    cbar=True,
    ax=ax
)
ax.set_title("Confusion matrix for TabTransformer (row-normalized)", pad=14)
ax.set_xlabel("Predicted class")
ax.set_ylabel("True class")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)

cbar = hm.collections[0].colorbar
cbar.set_label("Fraction of true-class samples", rotation=270, labelpad=22, fontsize=15)
cbar.ax.tick_params(labelsize=12)

plt.tight_layout()
plt.savefig("./img_tabtransformer/conf_matrix_row_normalized_PRD.png", dpi=600, bbox_inches="tight")
plt.savefig("./img_tabtransformer/conf_matrix_row_normalized_PRD.pdf", bbox_inches="tight")
plt.close()

# Raw-count version for appendix / checking
fig, ax = plt.subplots(figsize=(13, 11))
hm = sns.heatmap(
    conf_mat,
    cmap="Blues",
    square=True,
    xticklabels=display_labels,
    yticklabels=display_labels,
    linewidths=0.05,
    linecolor="lightgray",
    cbar=True,
    ax=ax
)
ax.set_title("Confusion matrix for TabTransformer (raw counts)", pad=14)
ax.set_xlabel("Predicted class")
ax.set_ylabel("True class")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)

cbar = hm.collections[0].colorbar
cbar.set_label("Count", rotation=270, labelpad=18, fontsize=15)
cbar.ax.tick_params(labelsize=12)

plt.tight_layout()
plt.savefig("./img_tabtransformer/conf_matrix_counts_PRD.png", dpi=600, bbox_inches="tight")
plt.savefig("./img_tabtransformer/conf_matrix_counts_PRD.pdf", bbox_inches="tight")
plt.close()






# === Export clean TabTransformer inference model, run Captum, save CSVs, and compare to TreeSHAP ===
import os, json, numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from captum.attr import IntegratedGradients, DeepLift
from scipy.stats import spearmanr
from typing import Optional

os.makedirs("./img_tabtransformer", exist_ok=True)

# ---------- 1) Export a clean inference model with frozen normalization ----------
scaler = StandardScaler().fit(train_df[num_features].to_numpy(dtype=np.float32))

class _LightningWrapper(nn.Module):
    """
    Wrapper for TabTransformer LightningModule.
    Prepares dict input {continuous, categorical} as TabTransformer expects.
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

# --- CHANGE: Force LightningWrapper for TabTransformer ---
def build_exported_model(tabular_model, scaler, device="cpu"):
    lm = tabular_model.model
    mu = scaler.mean_
    sigma = scaler.scale_
    print("[ExportedTabTransformer] Using LightningWrapper with dict input")
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

with open("./img_tabtransformer/attr_metadata_tabtransformer.json", "w") as f:
    json.dump({
        "model": "TabTransformer",
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

per_class_path = "./img_tabtransformer/captum_ig_per_class.csv"
global_path    = "./img_tabtransformer/captum_ig_global.csv"
per_class_df.to_csv(per_class_path, index=False)
global_df.to_csv(global_path, index=False)
print(f"Saved: {per_class_path}, {global_path}")

# ---------- 6) Compare to XGBoost TreeSHAP ----------
xgb_global_csv = "./img_xgboost/treesap_global.csv"
xgb_per_class_csv = "./img_xgboost/treesap_per_class.csv"

xgb_g = pd.read_csv(xgb_global_csv).set_index("feature")
tabtransformer_g = pd.read_csv(global_path).set_index("feature")
common_feats = xgb_g.index.intersection(tabtransformer_g.index)

rho_g, p_g = spearmanr(
    xgb_g.loc[common_feats, "mean_abs_shap"].rank(ascending=False),
    tabtransformer_g.loc[common_feats, "mean_abs_attr"].rank(ascending=False),
)
print(f"[GLOBAL] Spearman ρ(TreeSHAP vs Captum IG, TabTransformer) = {rho_g:.3f} (p={p_g:.3g})")

try:
    xgb_c = pd.read_csv(xgb_per_class_csv)
    tabtransformer_c = pd.read_csv(per_class_path)
    classes_to_check = sorted(set(xgb_c["class"]).intersection(set(tabtransformer_c["class"])))
    for cname in classes_to_check:
        xx = xgb_c[xgb_c["class"] == cname].set_index("feature")
        mm = tabtransformer_c[tabtransformer_c["class"] == cname].set_index("feature")
        feats = xx.index.intersection(mm.index)
        if len(feats) < 2: continue
        rho, p = spearmanr(
            xx.loc[feats, "mean_abs_shap"].rank(ascending=False),
            mm.loc[feats, "mean_abs_attr"].rank(ascending=False),
        )
        print(f"[CLASS {cname}] Spearman ρ = {rho:.3f} (p={p:.3g})")
except FileNotFoundError:
    print("Per-class comparison skipped: ./img_xgboost/treesap_per_class.csv not found.")




# === Export Correlation Results to CSV (TabTransformer) ===
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
        mm = tabtransformer_c[tabtransformer_c["class"] == cname].set_index("feature")
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
out_path = "./img_tabtransformer/captum_vs_treeshap_corr.csv"
corr_df.to_csv(out_path, index=False)
print(f"Saved correlations CSV: {out_path}")