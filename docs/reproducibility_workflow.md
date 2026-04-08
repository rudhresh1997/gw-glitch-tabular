# Reproducibility Workflow

This document summarizes the computational workflow used in the paper.

## Datasets

Two cleaned tabular datasets are used:

1. `original_data_cleaned.csv`  
   Full cleaned dataset with 500,000 examples.

2. `sampled_data_cleaned.csv`  
   Sampled cleaned dataset with 50,000 examples, constructed to preserve the class distribution, including the class imbalance structure, of the full dataset.

Each dataset contains:
- 9 feature columns
- 1 target column: `ml_label`

The feature definitions correspond to Table I of the paper.

## Data split

The train/validation/test split is:

- Train: 64%
- Validation: 16%
- Test: 20%

The workflow is intended to preserve class balance structure through stratified splitting.

## Step 1: Class-distribution analysis

The full dataset is used to generate the log-scale class distribution shown in Figure 1.

Supporting notebook:
- `notebooks/class_distribution_logscale.ipynb`

Target reproducible script:
- `src/paper_figures/fig1_class_distribution.py`

## Step 2: Hyperparameter tuning on sampled dataset

Hyperparameter tuning is performed on `sampled_data_cleaned.csv` for computational efficiency while preserving the original class distribution.

For each model:
- 5-fold cross-validation is used
- 100 Optuna trials are performed
- the best model is selected using weighted F1 score
- average training time is recorded
- average inference (test) time is recorded

Models included:
- XGBoost
- MLP
- TabNet
- TabTransformer
- FT-Transformer
- DANet
- AutoInt
- GATE
- GANDALF
- NODE

## Step 3: Final training on full dataset

Using the best hyperparameters from the tuning stage, each model is trained on `original_data_cleaned.csv`.

For each model:
- training is repeated across 15 random seeds
- final performance is summarized using classification reports and confusion matrices

## Step 4: Evaluation outputs

Each model produces evaluation outputs including:
- confusion matrix
- `metrics.csv`
- PR curve
- ROC curve

The `metrics.csv` file stores:
- per-class F1 scores
- average weighted F1 score
- training time
- test/inference time

## Step 5: Interpretability analysis

Interpretability outputs depend on model class.

### XGBoost
TreeSHAP is used to generate:
- `treeshap_per_class.csv`
- `treeshap_global.csv`

### Neural and deep tabular models
Captum Integrated Gradients is used to generate:
- `captum_ig_per_class.csv`
- `captum_ig_global.csv`

In addition, each non-XGBoost model is compared against the XGBoost TreeSHAP baseline using Spearman rank correlation:
- `captum_vs_treeshap_corr.csv`

## Step 6: Cross-model summary files

Lightweight summary CSV files are assembled across models to generate comparative plots and interpretability analyses:

- `f1_scores_all_models.csv` — Figure 3
- `training_time_summary.csv` — Figure 4
- `inference_time_summary.csv` — Figure 5
- `model_complexity_summary.csv` — Figure 6
- `spearman_vs_xgboost.csv` — Figure 7
- `cross_model_alignment.csv` — Figure 8
- `normalized_confusion_matrix_danet.csv` — Figure 9
- `counts_confusion_matrix_danet.csv` — Figure 10
- `scaling_behavior.csv` — Figure 11
- `global_feature_importance_summary.csv` — Figure 12
- `modelwise_feature_importance_summary.csv` — Figure 13

## Step 7: Figure generation

The main paper figures are generated from summary CSV files and model-specific outputs.

Figure/script mapping is documented in:
- `paper_assets/figure_to_script_map.md`

## Legacy code note

The repository preserves original research scripts under `src/legacy/` while modular refactoring proceeds into separate tuning, training, evaluation, interpretability, and figure-generation modules.

For several PyTorch Tabular models, both an original script and a refined `v2` version are preserved. The `v2` versions typically correspond to improved working variants, while earlier versions are retained for provenance.