# glitch-tabular-prd

Code and reproducibility package for the paper:

**Evaluating Deep Learning Models for Multiclass Classification of LIGO Gravitational Wave Glitches**

## Overview

This repository contains code, configurations, summary results, interpretability analyses, and figure-generation assets for benchmarking classical and deep learning models on tabular glitch metadata.

## Models

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

## Dataset

Two cleaned datasets are used locally during the workflow:

- `original_data_cleaned.csv`: full dataset with 500,000 samples
- `sampled_data_cleaned.csv`: sampled dataset with 50,000 samples preserving the original class distribution

Each dataset contains:
- 9 feature columns
- 1 target column: `ml_label`

The feature definitions correspond to Table I of the paper.

## Split policy

Train/validation/test split:
- Train: 64%
- Validation: 16%
- Test: 20%

## Workflow

1. Generate class-distribution figure from the full dataset
2. Tune models on the sampled dataset using 5-fold cross-validation and 100 Optuna trials
3. Train final models on the full dataset using best hyperparameters across 15 random seeds
4. Evaluate models and save metrics, confusion matrices, ROC curves, and PR curves
5. Compute interpretability outputs using TreeSHAP or Captum
6. Generate paper figures from summary CSV files and model-specific outputs

## Summary files used for paper figures

The `results/summary/` directory contains the lightweight CSV files used to generate the main figures in the paper:

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

## Legacy scripts

Original research scripts are preserved under `src/legacy/` during modular refactoring.

For several PyTorch Tabular models, both an original script and a refined `v2` script are preserved. For example:
- `danet.py` and `danetv2.py`
- `danet.pbs` and `danetv2.pbs`

The `v2` scripts generally correspond to improved working variants, while the earlier versions are retained for provenance.

## Repository status

This repository is being prepared as the public companion repository for paper reproducibility and DOI archiving.