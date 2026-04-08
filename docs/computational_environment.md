# Computational Environment

This document summarizes the computational environments used in this project.

## Local repository organization

Repository organization, restructuring, and lightweight analysis tasks were performed locally on a MacBook Air (2021).

## Primary training environment

Primary model training and large-scale experiments were conducted on the Kodiak computing environment.

The project used GPU-queue execution for:
- XGBoost
- MLP
- PyTorch Tabular models

HPC submission scripts are stored in:
- `scripts/hpc/`

## Software stack

The project uses Python-based machine learning and scientific-computing tools, including:

- Python
- NumPy
- pandas
- scikit-learn
- matplotlib
- Optuna
- XGBoost
- PyTorch
- PyTorch Tabular
- SHAP
- Captum

Environment files:
- `requirements.txt`
- `environment.yml`

These will be finalized to reflect the exact reproducibility environment for public release and DOI archiving.

## Experimental setup summary

- Hyperparameter tuning was performed on the sampled 50,000-example dataset.
- Final multi-seed training was performed on the full 500,000-example dataset.
- The train/validation/test split was 64/16/20.
- Tuning used 5-fold cross-validation with 100 Optuna trials.
- Final training used 15 random seeds per model.

## Reproducibility note

This repository is being prepared as the paper companion repository for public reproducibility release. The current structure separates legacy research scripts from modularized reproducibility scripts.

## Legacy execution variants

For several PyTorch Tabular models, both original and refined `v2` execution variants are preserved in the repository:

- original Python scripts are stored in `src/legacy/`
- corresponding HPC submission scripts are stored in `scripts/hpc/`

For example, `danet.py` and `danetv2.py` are paired with `danet.pbs` and `danetv2.pbs`, respectively. The `v2` variants typically reflect improved working versions used during later experimentation.