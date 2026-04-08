# Split Manifest

The project uses a train/validation/test split of:

- Train: 64%
- Validation: 16%
- Test: 20%

The split is intended to preserve the class-distribution structure of the dataset.

Target column:
- `ml_label`

This manifest documents the intended split policy for reproducibility. If explicit split-index files are later added, they should be stored in this directory.