# JPCO O3 dataset and split package v1.0

This package defines a corrected benchmark for JPCO-103168. It does not contain newly trained models or new performance claims. Original source files and previously reported results are not overwritten.

## Inputs and provenance

Inputs are H1_O3a.csv, H1_O3b.csv, L1_O3a.csv and L1_O3b.csv supplied by the author. O1 and O2 are deliberately excluded to retain the submitted O3 scope. The manifest records source SHA-256 hashes and software versions.

`o3_master.csv.gz` retains all 500,524 O3 source records, the eleven numerical features, labels, detector, source catalog, zero-based source data-row index, Gravity Spy ID, source confidence, channel and image URLs. Source class-probability columns are intentionally omitted. This is a selected-column provenance master, not a verbatim copy of the catalogs. Missing source image metadata remains missing; URLs have not been tested or downloaded.

`o3_primary.csv.gz` is a smaller model-input table. `source_record_id` joins each representative back to the master. The master maps repeated source rows to their representative. Read these compressed CSVs directly with pandas; decompression to disk is unnecessary.

## Event grouping and label policy

Rows are linked if they share (1) detector and exact integer peak GPS nanoseconds, (2) a Gravity Spy ID, or (3) all eleven numerical features, even across detectors. Connected components define `group_id`. The third link is a conservative safeguard against identical representations crossing partitions. Group IDs are record-group proxies, not a claim that all nearby glitches have been physically identified as one event.

All 18 components with inconsistent labels (39 source records) are quarantined from the primary benchmark. Their records and label summaries are preserved in `conflicting_records.csv` and `conflicting_groups.csv`. No majority vote, confidence selection or manual relabeling is applied. Seventeen conflicting detector/time groups share the same eleven-feature vector within each group; one group contains two distinct vectors. Source metadata alone cannot determine the correct label. A future adjudication should be versioned, not silently added to these frozen splits.

Within nonconflicting groups, exact duplicates of detector + eleven features + label are collapsed to the first source record in the fixed source order. Distinct feature vectors are retained and share a group. Collapsing copies reduces their evaluation weight; this is intentionally a corrected evaluation population. It does not establish that the source records were created in error. No original source row is deleted from the master.

## Outer split specifications

- **Grouped random:** split group IDs with stratification by label and seed 42 into approximately 64% train, 16% validation and 20% test. Fractions apply to group counts; row fractions can differ slightly. All retained records in a component stay together.
- **Chronological:** choose cutoffs at the 64% and 80% ordered primary-row positions using exact integer GPS nanoseconds. Entire components go to partitions based on their latest timestamp; tied timestamps go to the later partition. Strict train-before-validation-before-test ordering is asserted. If a future catalog change introduces a component spanning a cutoff, the build rejects any resulting temporal-order violation.

`split_assignments.csv.gz` lists primary representatives and both assignments. Excluded groups receive `excluded_conflicting_label` in the master. Repeated nonconflicting copies inherit their representative group's assignments but are not part of primary scoring. Do not combine the master and primary table as training rows.

The random and chronological configurations are alternative experiments. Their partitions overlap across configurations by design. Do not use test scores from either configuration to choose preprocessing, hyperparameters or which result to report. Freeze the analysis protocol first and report both conditions.

## Tuning and preprocessing

Optional `tuning_random_split.csv.gz` and `tuning_chronological_split.csv.gz` contain approximately 32,000 rows each, chosen as complete label-stratified groups strictly within their respective outer training partitions (seed 20260915). They replace the historical globally sampled tuning file. They are tuning-membership lists, not inner-CV folds.

For random experiments use grouped inner cross-validation. For temporal experiments use forward-in-time inner validation and check rare-class coverage in each fold. Group-aware stratification can consult training labels; never use outer validation/test outcomes to design folds. The inner-CV plan and trial budget still need to be fixed during pilot preparation. Fit all scalers and transformations on each training fold only. Compute class weights from the corresponding training data only. Final early stopping may use outer validation; test remains untouched until final evaluation.

The chronological data naturally omit some rare labels from later partitions. Report class support, all-class and supported-class metric conventions explicitly. Do not move later records backward or rebalance test data to enforce class presence. These are catalog ML labels; this package does not verify human ground truth.

## Predictor allowlists

Use `manifest.json` feature lists explicitly. NEVER construct X by dropping only `ml_label` from these tables: provenance, group IDs, split names and timestamps used solely for partitioning must not become unintended predictors.

- Original nine: peak_time, peak_time_ns, start_time_ns, duration, peak_frequency, central_freq, amplitude, snr, q_value.
- Timestamp-free six: duration, peak_frequency, central_freq, amplitude, snr, q_value.
- Enriched eleven: the original nine plus start_time and bandwidth. This is an optional scalar-feature ablation, not a substitute for waveform-derived features.

```python
import json
import pandas as pd
from pathlib import Path

base = Path('jpco_o3_splits_v1')
spec = json.loads((base / 'manifest.json').read_text())
features = spec['feature_sets']['timestamp_free_six']
cols = features + ['ml_label', 'group_id', 'peak_gps_ns', 'random_split', 'chronological_split']
data = pd.read_csv(base / 'o3_primary.csv.gz', usecols=cols)
protocol = 'chronological_split'
train = data.loc[data[protocol].eq('train')]
validation = data.loc[data[protocol].eq('validation')]
test = data.loc[data[protocol].eq('test')]
X_train, y_train = train[features], train['ml_label']
```

For the 8 GB Mac, begin with a single CPU XGBoost pilot and a single feature configuration. Load only needed columns, use one training process, and measure runtime and memory before scheduling additional fits. A runtime budget has not yet been measured on that Mac.

## Validation and limitations

The builder asserts zero group overlap and zero exact nine-feature overlap across every pair of partitions within each protocol. It also asserts strict chronological ordering and training-only tuning membership. `split_summary.csv`, `class_counts.csv` and `manifest.json` record the resulting counts and checks.

These checks address identified repeated-row and tuning contamination. They do not prove statistical independence of nearby events or detector states. No temporal embargo or broader clustering window is applied in v1. A temporal-gap sensitivity experiment may still be appropriate. Exact historical source-to-processed numerical reconstruction is not completed; v1 uses catalog values directly. Previously observed CSV serialization differences are not patched into catalog values.

## Reproduce

With the four original catalogs in one directory:

```bash
python build_splits.py --input-dir /path/to/catalogs --output-dir ./rebuilt
```

Use the Python, NumPy, pandas and scikit-learn versions recorded in `manifest.json` to reproduce the split memberships. Input hashes must match. Source CSVs are not included in this archive. `SHA256SUMS.txt` verifies the package members before extraction or use. The prebuilt files are ready to use; rebuilding is optional.
