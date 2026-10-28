# gw-glitch-tabular

Code and reproducibility materials for:

**Evaluating Machine-Learning Models for Multiclass Classification of
LIGO Gravitational-Wave Glitches**

Zenodo DOI:
https://doi.org/10.5281/zenodo.19475319

## Overview

This repository contains code, configurations, lightweight summary
results, interpretability analyses, and figure-generation assets for
benchmarking classical and deep-learning models on tabular gravitational-
wave glitch metadata.

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

The source data were obtained from the Gravity Spy O3 dataset:

https://doi.org/10.5281/zenodo.5649212

Raw datasets are not included because of their size and redistribution
considerations.

The original benchmark used two locally prepared cleaned datasets:

- `original_data_cleaned.csv`: 500,524 rows;
- `sampled_data_cleaned.csv`: 50,554 rows.

Each contains nine feature columns and one target column, `ml_label`.
The historical sampled dataset began with an approximately 50,000-row
stratified sample. Its four rarest classes were subsequently expanded
with replacement, producing the final total of 50,554 rows. See the
2026 revision audit for the precise provenance and limitations of this
historical procedure.

## Original benchmark workflow

The original benchmark used a stratified 64/16/20
train/validation/test division:

- training: 64%;
- validation: 16%;
- test: 20%.

The workflow was:

1. generate the class-distribution figure;
2. tune models on the sampled dataset using five-fold cross-validation
   and Optuna;
3. train final models on the full processed dataset across multiple
   random seeds;
4. evaluate classification and computational metrics;
5. compute model-native feature attributions using TreeSHAP or Captum;
6. generate manuscript figures from lightweight summary files.

## 2026 referee-driven revision package

The curated revision package is located at
revisions/jpco_2026/.

It contains:

- six portable, output-cleared notebooks;
- reviewed experiment and audit scripts;
- a duplicate-aware O3 cohort and split protocol;
- grouped-random and chronological generalization analyses;
- timestamp and targeted feature ablations;
- native-attribution and common permutation-importance analyses;
- operational footprint and CPU-latency measurements;
- bandwidth-augmentation results;
- feature-provenance and image-comparison feasibility audits; and
- lightweight, checksum-audited numerical evidence.

The reconstructed revision cohort contains 437,753 events. Identity
groups were kept disjoint across frozen 64/16/20 grouped-random and
chronological partitions. Large event-level datasets, trained models,
predictions, checkpoints, and logs are intentionally excluded.

For scope, environment, path configuration, execution order, and
reproducibility boundaries, see
revisions/jpco_2026/README.md.

## Summary files for the original paper figures

The `results/summary/` directory contains lightweight CSV files used to
generate the original manuscript figures, including:

- `f1_scores_all_models.csv`
- `training_time_summary.csv`
- `inference_time_summary.csv`
- `model_complexity_summary.csv`
- `spearman_vs_xgboost.csv`
- `cross_model_alignment.csv`
- `normalized_confusion_matrix_danet.csv`
- `counts_confusion_matrix_danet.csv`
- `scaling_behavior.csv`
- `global_feature_importance_summary.csv`
- `modelwise_feature_importance_summary.csv`

The revised operational-complexity Figure 6 and its source table are
distributed within the revision evidence directory.

## Legacy scripts

Original research scripts are preserved under `src/legacy/` during
modular refactoring. Where both original and `v2` variants exist, the
earlier scripts are retained for provenance and the later variants
generally represent refined working implementations.

## Repository status

This repository accompanies the article accepted for publication in
*Journal of Physics Communications*.

Article DOI: https://doi.org/10.1088/2399-6528/aeb8df

The final Version of Record may not yet be available.

## Citation

If you use this repository, cite the archived software release using the
Zenodo DOI above. Citation metadata are provided in `CITATION.cff`.
