"""JPCO MLP resource pilot: full grouped-random training, six features, three epochs.

Run as a fresh env1 subprocess. No installation, tuning, test evaluation, or
checkpoint selection. This script never imports or executes legacy mlp3.py.

ARCHIVAL AUDIT (September 16, 2026)
The supplied mlp.pbs invokes mlp3.py, and the error log names that script.
All 100 Optuna trials completed; trial 78 won with CV macro F1 0.4457163058.
The legacy script is custom PyTorch, not PyTorch Tabular. It tunes on the
whole sampled CSV using shuffled stratified 3-fold CV, five epochs per fold,
unweighted cross entropy, StandardScaler fitted to each training fold, and
AdamW. Final fits use 20 epochs and batch size 1024. The 15 values 42..56
seed data splits; they do not explicitly seed PyTorch initialization/dropout
or Optuna. Validation data are created but unused for stopping/selection.
The output log reports mean weighted F1 0.8194 +/- 0.0032; the script uses
population SD (ddof=0). metrics.csv, plots and attribution use the run with
highest TEST F1, not validation-selected or seed-mean performance. No model
state or fitted scaler is saved by mlp3.py (only the label encoder is).
Integrated Gradients targets predicted-class logits and aggregates by
predicted class, using zero in standardized space as baseline. DeepLift is
instantiated but no DeepLift attribution is computed. Do not describe these
as true-label class groups or evidence of explanation-method robustness.

PILOT DESIGN
Retain the recovered [512,256,128] Mish topology only as a resource probe.
Use fixed diagnostic lr=0.001, dropout=0.1, AdamW weight_decay=0.01 and
unweighted cross entropy. Historical tuned values are NOT clean selections
for these new splits. The pilot configuration is not a final selected model.
Standardize float64 training values before converting to float32 tensors.
Scalers and labels are fitted to training only. Use seed42, two CPU threads,
one interop thread, no DataLoader workers, no accelerator, no early stopping.
This unweighted loss differs from the inverse-frequency-weighted XGBoost
revision objective; final methods must disclose model-specific objectives.

Official implementation references:
https://docs.pytorch.org/docs/stable/generated/torch.manual_seed.html
https://docs.pytorch.org/docs/stable/generated/torch.set_num_threads.html
https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html
"""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import resource
import sys
import time

for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[name] = '2'

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def peak_mib():
    v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(v / (1024**2 if sys.platform == 'darwin' else 1024))


def load_partitions(base):
    expected = {}
    for line in (base / 'SHA256SUMS.txt').read_text().splitlines():
        checksum, name = line.split(maxsplit=1)
        expected[name.strip()] = checksum
    names = ['manifest.json', 'o3_primary.csv.gz', 'split_summary.csv']
    hashes = {name: sha(base / name) for name in names}
    for name in names:
        if hashes[name] != expected.get(name):
            raise ValueError(f'Checksum mismatch: {name}')
    spec = json.loads((base / 'manifest.json').read_text())
    features = spec['feature_sets']['timestamp_free_six']
    assert features == ['duration', 'peak_frequency', 'central_freq', 'amplitude', 'snr', 'q_value']
    cols = ['source_record_id', 'group_id', 'random_split', 'ml_label'] + features
    train_parts, val_parts = [], []
    for chunk in pd.read_csv(base / 'o3_primary.csv.gz', usecols=cols,
                             chunksize=50000, float_precision='round_trip'):
        # Discard test rows while scanning the common source file.
        train_parts.append(chunk.loc[chunk.random_split.eq('train')].copy())
        val_parts.append(chunk.loc[chunk.random_split.eq('validation')].copy())
    train = pd.concat(train_parts, ignore_index=True)
    val = pd.concat(val_parts, ignore_index=True)
    assert train.source_record_id.is_unique and val.source_record_id.is_unique
    assert set(train.group_id).isdisjoint(val.group_id)
    assert len(train) == 280163 and len(val) == 70040
    summary = pd.read_csv(base / 'split_summary.csv')
    for partition, frame in [('train', train), ('validation', val)]:
        expected_n = summary.loc[(summary.configuration == 'random_split') &
                                 (summary.partition == partition), 'rows'].item()
        assert len(frame) == expected_n
    assert train.ml_label.nunique() == val.ml_label.nunique() == 24
    assert set(val.ml_label) <= set(train.ml_label)
    return train, val, features, hashes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir', type=Path, required=True)
    ap.add_argument('--output-root', type=Path, required=True)
    args = ap.parse_args()
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as e:
        raise SystemExit(f'PyTorch could not import in {sys.executable}: {e}\n'
                         'Share this error; do not change env1 packages yet.')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    print(f'Python: {sys.executable}\nPyTorch: {torch.__version__}; device: CPU', flush=True)
    print('Checking input hashes and loading frozen training/validation partitions...', flush=True)
    train, val, features, hashes = load_partitions(args.data_dir.expanduser().resolve())
    encoder = LabelEncoder().fit(train.ml_label)
    y_train = encoder.transform(train.ml_label)
    y_val = encoder.transform(val.ml_label)
    train_values = train[features].to_numpy(dtype=np.float64)
    val_values = val[features].to_numpy(dtype=np.float64)
    assert np.isfinite(train_values).all() and np.isfinite(val_values).all()
    scaler = StandardScaler().fit(train_values)
    X_train = scaler.transform(train_values).astype(np.float32)
    X_val = scaler.transform(val_values).astype(np.float32)
    assert np.isfinite(X_train).all() and np.isfinite(X_val).all()
    val_ids = val.source_record_id.astype(str).to_numpy(dtype=str)
    del train, val, train_values, val_values
    gc.collect()

    out = args.output_root.expanduser().resolve() / datetime.now(timezone.utc).strftime('mlp_full_pilot_%Y%m%dT%H%M%S_%fZ')
    out.mkdir(parents=True, exist_ok=False)
    config = dict(purpose='Three-epoch resource pilot, not a selected final model',
                  features=features, hidden_dims=[512, 256, 128], activation='Mish',
                  dropout=0.1, learning_rate=0.001, weight_decay=0.01,
                  loss='unweighted cross entropy', batch_size=1024, epochs=3,
                  seed=42, device='cpu', threads=2, interop_threads=1,
                  split='random_split', test_evaluated=False,
                  train_rows=len(y_train), validation_rows=len(y_val),
                  labels=encoder.classes_.tolist(), input_sha256=hashes,
                  script_sha256=sha(Path(__file__)),
                  scaling='training-fitted StandardScaler in float64, then float32 tensors',
                  versions=dict(python=platform.python_version(), torch=str(torch.__version__),
                                numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__),
                  platform=platform.platform(), architecture=platform.machine())
    save_json(out / 'pilot_config.json', config)
    save_json(out / 'scaler.json', dict(features=features, mean=scaler.mean_.tolist(),
                                     scale=scaler.scale_.tolist(), var=scaler.var_.tolist(),
                                     n_samples_seen=int(scaler.n_samples_seen_)))
    layers = []
    previous = len(features)
    for width in config['hidden_dims']:
        layers.extend([nn.Linear(previous, width), nn.Mish(), nn.Dropout(config['dropout'])])
        previous = width
    layers.append(nn.Linear(previous, 24))
    model = nn.Sequential(*layers).cpu()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'],
                                 weight_decay=config['weight_decay'])
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
                        batch_size=1024, shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(42))
    val_tensor = torch.from_numpy(X_val)
    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Train: {len(y_train):,}; validation: {len(y_val):,}; parameters: {parameter_count:,}', flush=True)
    print('Three fixed epochs; two CPU threads; batch size 1024; no early stopping.', flush=True)
    history = []
    before = peak_mib()
    t0 = time.perf_counter()
    for epoch in range(1, 4):
        te = time.perf_counter()
        model.train()
        loss_sum = 0.0
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            if not torch.isfinite(loss):
                raise ValueError('Non-finite training loss; share the error.')
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(yb)
        row = dict(epoch=epoch, train_loss=loss_sum/len(y_train), seconds=time.perf_counter()-te)
        history.append(row)
        print(f"Epoch {epoch}/3: loss={row['train_loss']:.6f}; {row['seconds']:.2f} seconds", flush=True)
    fit_seconds = time.perf_counter() - t0
    after_fit = peak_mib()
    pd.DataFrame(history).to_csv(out / 'training_history.csv', index=False)
    model.eval()
    t0 = time.perf_counter()
    with torch.inference_mode():
        probs = np.concatenate([model(x).softmax(dim=1).numpy() for x in val_tensor.split(1024)])
    predict_seconds = time.perf_counter() - t0
    assert probs.shape == (len(y_val), 24) and np.isfinite(probs).all()
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
    pred = probs.argmax(axis=1)
    torch.save(model.state_dict(), out / 'pilot_state_dict.pt')
    np.savez_compressed(out / 'validation_predictions.npz', source_record_id=val_ids,
                        y_true=y_val, y_pred=pred, probabilities=probs)
    result = dict(train_rows=len(y_train), validation_rows=len(y_val), feature_count=6,
                  epochs=3, trainable_parameters=parameter_count, fit_seconds=fit_seconds,
                  epoch_seconds=[r['seconds'] for r in history],
                  validation_prediction_batch_seconds=predict_seconds,
                  validation_amortized_microseconds_per_row=predict_seconds/len(y_val)*1e6,
                  validation_weighted_f1=float(f1_score(y_val, pred, average='weighted', zero_division=0)),
                  validation_macro_f1_all24=float(f1_score(y_val, pred, labels=np.arange(24), average='macro', zero_division=0)),
                  validation_accuracy=float(accuracy_score(y_val, pred)),
                  process_peak_rss_before_fit_mib=before, process_peak_rss_after_fit_mib=after_fit,
                  process_peak_rss_mib=peak_mib(),
                  serialized_weights_mib=(out / 'pilot_state_dict.pt').stat().st_size/1024**2,
                  test_evaluated=False, output_directory=str(out),
                  elapsed_seconds_before_summary=time.perf_counter()-started,
                  interpretation='Fixed three-epoch resource pilot; not a final score or a convergence assessment.',
                  timing_note='CPU fit includes batch loading. Prediction covers all validation rows in minibatches, without a separate warm-up; not single-event latency.',
                  memory_note='Fresh-process peak RSS includes imports, training arrays, optimizer and preparation; not model-only or total system memory.',
                  size_note='Weights only; preprocessing and configuration are saved separately. Not a cross-family deployment-size comparison.')
    save_json(out / 'pilot_summary.json', result)
    print('\nPILOT COMPLETE\n' + json.dumps(result, indent=2), flush=True)
    print('\nShare pilot_summary.json and pilot_config.json before launching a search.', flush=True)


if __name__ == '__main__':
    main()
