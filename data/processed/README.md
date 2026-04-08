# data/processed

This directory stores cleaned datasets used by the training pipeline.

Local project datasets:
- `original_data_cleaned.csv` — full cleaned dataset with 500,000 samples
- `sampled_data_cleaned.csv` — sampled cleaned dataset with 50,000 samples preserving the class distribution of the full dataset

Each dataset contains:
- 9 feature columns
- 1 target column: `ml_label`

These files are excluded from Git tracking by default.