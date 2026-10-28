"""JPCO resource pilot: fixed training subset, six features, validation only.

Run with env1 Python as a subprocess from Jupyter so memory refers to a fresh process.
No package installation, hyperparameter search, early stopping or test scoring.
XGBoost parameters: https://xgboost.readthedocs.io/en/stable/parameter.html
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time
from datetime import datetime, timezone

# Set before numerical-library imports in this fresh process.
for variable in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[variable] = '2'

import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def peak_rss_mib():
    # macOS returns bytes; Linux returns KiB. This is process peak RSS, not model memory.
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(value / (1024**2 if sys.platform == 'darwin' else 1024))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    data_dir = args.data_dir.expanduser().resolve()
    checksums = {}
    for line in (data_dir / 'SHA256SUMS.txt').read_text().splitlines():
        checksum, filename = line.split(maxsplit=1)
        checksums[filename.strip()] = checksum
    inputs = ['manifest.json', 'o3_primary.csv.gz', 'tuning_random_split.csv.gz', 'split_summary.csv']
    print('Checking input hashes...', flush=True)
    used_hashes = {}
    for name in inputs:
        actual = digest(data_dir / name)
        if actual != checksums.get(name):
            raise ValueError(f'Checksum mismatch: {name}. Restore the file from the original ZIP.')
        used_hashes[name] = actual
    spec = json.loads((data_dir / 'manifest.json').read_text())
    features = spec['feature_sets']['timestamp_free_six']
    expected_features = ['duration','peak_frequency','central_freq','amplitude','snr','q_value']
    assert features == expected_features, 'Unexpected feature specification'
    members = pd.read_csv(data_dir / 'tuning_random_split.csv.gz')
    assert members.source_record_id.is_unique
    wanted_ids = set(members.source_record_id)
    wanted_groups = set(members.group_id)
    columns = ['source_record_id','group_id','random_split','ml_label'] + features
    train_chunks, val_chunks = [], []
    print('Loading frozen training subset and validation partition...', flush=True)
    # The file is scanned in chunks; test rows are discarded, never fitted or scored.
    for chunk in pd.read_csv(data_dir / 'o3_primary.csv.gz', usecols=columns,
                             float_precision='round_trip', chunksize=50000):
        selected = chunk.source_record_id.isin(wanted_ids)
        assert chunk.loc[selected,'random_split'].eq('train').all()
        # Ensure tuning contains whole groups, including groups with multiple representatives.
        assert chunk.group_id.isin(wanted_groups).equals(selected)
        if selected.any():
            train_chunks.append(chunk.loc[selected].copy())
        val_mask = chunk.random_split.eq('validation')
        if val_mask.any():
            val_chunks.append(chunk.loc[val_mask].copy())
    train = pd.concat(train_chunks, ignore_index=True)
    val = pd.concat(val_chunks, ignore_index=True)
    del train_chunks, val_chunks, chunk
    assert set(train.source_record_id) == wanted_ids
    assert len(train) == spec['tuning_subsets']['random_split']['rows'] == 32000
    assert set(train.group_id).isdisjoint(set(val.group_id))
    summary = pd.read_csv(data_dir / 'split_summary.csv')
    expected_val = int(summary.loc[(summary.configuration=='random_split') &
                                  (summary.partition=='validation'),'rows'].iloc[0])
    assert len(val) == expected_val == 70040
    assert train.ml_label.nunique() == val.ml_label.nunique() == 24
    assert set(val.ml_label) <= set(train.ml_label)
    encoder = LabelEncoder().fit(train.ml_label)
    y_train = encoder.transform(train.ml_label)
    y_val = encoder.transform(val.ml_label)
    X_train = train[features].astype(np.float32)
    X_val = val[features].astype(np.float32)
    assert np.isfinite(X_train.to_numpy()).all() and np.isfinite(X_val.to_numpy()).all()
    # Normalize inverse-frequency weights to mean one. Training labels only.
    # This pilot weighting is not an exact reproduction of the legacy N/n_class weights.
    weights = compute_sample_weight('balanced', y_train).astype(np.float32)
    assert np.isclose(weights.mean(), 1.0)
    class_counts = train.ml_label.value_counts().sort_index().to_dict()
    del train, val, members, wanted_ids, wanted_groups
    gc.collect()
    params = dict(n_estimators=100, max_depth=6, learning_rate=0.1, max_bin=256,
                  min_child_weight=1, subsample=1.0, colsample_bytree=1.0,
                  reg_lambda=1.0, reg_alpha=0.0, tree_method='hist', device='cpu',
                  objective='multi:softprob', num_class=len(encoder.classes_),
                  eval_metric='mlogloss', n_jobs=2, random_state=42, verbosity=1)
    run_name = datetime.now(timezone.utc).strftime('xgb_pilot_%Y%m%dT%H%M%S_%fZ')
    out = args.output_root.expanduser().resolve() / run_name
    out.mkdir(parents=True, exist_ok=False)
    config = dict(purpose='Resource pilot, validation only; not a final benchmark',
                  protocol='random_split', features=features, params=params,
                  sample_weight_rule='N_train / (n_classes * count_in_training_class)',
                  train_rows=len(y_train), validation_rows=len(y_val),
                  training_class_counts={str(k):int(v) for k,v in class_counts.items()},
                  labels=encoder.classes_.tolist(), input_sha256=used_hashes,
                  versions=dict(python=platform.python_version(), numpy=np.__version__,
                                pandas=pd.__version__, sklearn=sklearn.__version__, xgboost=xgb.__version__),
                  platform=platform.platform(), architecture=platform.machine(),
                  cpu_logical_count=os.cpu_count(), test_evaluated=False)
    (out/'pilot_config.json').write_text(json.dumps(config, indent=2)+'\n')
    print(f'Train: {len(y_train):,}; validation: {len(y_val):,}; features: {len(features)}', flush=True)
    print('CPU threads: 2; rounds: 100; maximum depth: 6; no early stopping.', flush=True)
    print('Training starts; progress prints every 20 rounds.', flush=True)
    model = xgb.XGBClassifier(**params)
    peak_before = peak_rss_mib()
    t0 = time.perf_counter()
    # eval_set reports loss only; it does not change the fixed 100-round fit.
    model.fit(X_train, y_train, sample_weight=weights,
              eval_set=[(X_val, y_val)], verbose=20)
    fit_seconds = time.perf_counter()-t0
    peak_after_fit = peak_rss_mib()
    t0=time.perf_counter()
    probabilities=model.predict_proba(X_val)
    prediction_seconds=time.perf_counter()-t0
    assert probabilities.shape == (len(y_val),len(encoder.classes_))
    assert np.isfinite(probabilities).all()
    prediction=np.argmax(probabilities,axis=1)
    model_path=out/'pilot_model.ubj'
    model.save_model(model_path)
    report=classification_report(y_val,prediction,labels=np.arange(len(encoder.classes_)),
                                 target_names=encoder.classes_.tolist(),zero_division=0,output_dict=True)
    (out/'validation_classification_report.json').write_text(json.dumps(report,indent=2)+'\n')
    pd.DataFrame(model.evals_result()['validation_0']).rename_axis('boosting_round').to_csv(out/'validation_loss.csv')
    results=dict(train_rows=len(y_train),validation_rows=len(y_val),feature_count=len(features),
                 fit_seconds=fit_seconds,validation_predict_proba_batch_seconds=prediction_seconds,
                 validation_amortized_microseconds_per_row=prediction_seconds/len(y_val)*1e6,
                 validation_weighted_f1=float(f1_score(y_val,prediction,average='weighted',zero_division=0)),
                 validation_macro_f1_24_classes=float(f1_score(y_val,prediction,labels=np.arange(24),average='macro',zero_division=0)),
                 validation_accuracy=float(accuracy_score(y_val,prediction)),
                 process_peak_rss_before_fit_mib=peak_before,process_peak_rss_after_fit_mib=peak_after_fit,
                 process_peak_rss_mib=peak_rss_mib(),serialized_model_mib=model_path.stat().st_size/1024**2,
                 elapsed_seconds_before_summary=time.perf_counter()-started,
                 test_evaluated=False,output_directory=str(out),
                 interpretation='Fixed-configuration pilot on 32k training rows; not a final score or a full-workload runtime estimate.',
                 timing_note='Fit includes validation loss evaluation each round. Prediction is one un-warmed full-validation batch; amortized time is not single-event latency.',
                 memory_note='Peak RSS of this fresh Python process, including imports and data preparation; not model-only memory or total system memory.')
    (out/'pilot_summary.json').write_text(json.dumps(results,indent=2)+'\n')
    print('\nPILOT COMPLETE',flush=True)
    print(json.dumps(results,indent=2),flush=True)
    print('\nStop after this pilot and share pilot_summary.json. Do not launch a parameter search yet.',flush=True)


if __name__=='__main__':
    main()
