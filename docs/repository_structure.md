# Repository Structure

This repository is the reproducibility package for the paper:

**Evaluating Deep Learning Models for Multiclass Classification of LIGO Gravitational Wave Glitches**

## Top-level structure

- `configs/`  
  Model- and data-specific configuration files.

- `data/`  
  Dataset documentation, processed local datasets, and split metadata.

- `docs/`  
  Documentation for repository organization, computational workflow, and environment.

- `figures/`  
  Final paper figures in PNG/PDF format.

- `notebooks/`  
  Supporting notebooks used during analysis and figure preparation.

- `paper_assets/`  
  Assets directly supporting the paper, including figure-to-script mapping and data-availability text.

- `results/`  
  Model-specific outputs and summary CSV files used for cross-model comparisons and figure generation.

- `scripts/`  
  Shell scripts for running model pipelines and HPC submission scripts for Kodiak.

- `src/`  
  Core source code for preprocessing, tuning, training, evaluation, interpretability, and paper-figure generation.

## Source code layout

- `src/preprocessing/`  
  Dataset validation, split generation, and class-distribution summarization.

- `src/tuning/`  
  Hyperparameter optimization scripts for each model using the sampled dataset.

- `src/training/`  
  Final multi-seed training scripts using the full dataset.

- `src/evaluation/`  
  Model evaluation scripts and aggregation of metrics.

- `src/interpretability/`  
  TreeSHAP- and Captum-based feature-importance analyses, including alignment comparisons.

- `src/paper_figures/`  
  Scripts for generating paper figures from summary CSV files and model outputs.

- `src/legacy/`  
  Original research scripts preserved during modular refactoring. In several cases, both an original script and a refined `v2` version are retained.

- `src/utils/`  
  Shared utility functions for paths, metrics, plotting, I/O, and random seeds.

## Results layout

Each model has a dedicated folder in `results/` with the following intended structure:

- `tuning/`  
  Hyperparameter-optimization outputs.

- `training/`  
  Outputs from final training runs.

- `metrics/`  
  Metrics tables and classification summaries.

- `plots/`  
  ROC curves, PR curves, and confusion matrices.

- `interpretability/`  
  Global and per-class feature-importance outputs.

- `checkpoints/`  
  Optional saved model checkpoints, excluded from version control unless lightweight.

## Summary outputs

The `results/summary/` directory contains lightweight cross-model CSV files used to generate the main comparative figures in the paper:

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

## Data note

Local datasets used during development are stored under `data/processed/` but are excluded from version control by default.