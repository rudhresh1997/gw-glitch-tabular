"""JPCO FT-Transformer CPU resource pilot, PyTorch Tabular 1.1.1.

Two fixed epochs on the frozen 32k grouped-random training subset; six
non-timestamp features; full outer validation; no test evaluation or tuning.
This is a resource probe, not a selected configuration or final score.
Run in a fresh env1 subprocess. Does not install packages or run legacy code.
"""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time

for variable in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[variable] = '2'

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler

EXPECTED = {
    'manifest.json': 'f1eedfaa282d6ae90e1567d271f9baff06c62ba233aff5890ae5dad85cfe9605',
    'o3_primary.csv.gz': '9d7d7d217563ccb121eba466fc442d83d7f25495e6e5ba40ea9a0962114829c6',
    'tuning_random_split.csv.gz': '8afbc1e03efa547ae4d51dcd3982308ebaa67f018f2fc225b59daaa55f153342',
    'split_summary.csv': '61de488a44179ca86205ffcb45e98baa9eb2d9a7b52a1ab13cf2845368f88d01',
}
FEATURES = ['duration', 'peak_frequency', 'central_freq', 'amplitude', 'snr', 'q_value']


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def peak_mib():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(value / (1024**2 if sys.platform == 'darwin' else 1024))


def load_data(base):
    hashes = {name: sha(base / name) for name in EXPECTED}
    if hashes != EXPECTED:
        raise ValueError('Frozen input hash mismatch; stop and share the error.')
    spec = json.loads((base / 'manifest.json').read_text())
    assert spec['feature_sets']['timestamp_free_six'] == FEATURES
    members = pd.read_csv(base / 'tuning_random_split.csv.gz')
    assert members.source_record_id.is_unique
    wanted_ids, wanted_groups = set(members.source_record_id), set(members.group_id)
    train_parts, val_parts = [], []
    cols = ['source_record_id', 'group_id', 'random_split', 'ml_label'] + FEATURES
    for chunk in pd.read_csv(base / 'o3_primary.csv.gz', usecols=cols,
                             float_precision='round_trip', chunksize=50000):
        selected = chunk.source_record_id.isin(wanted_ids)
        assert chunk.loc[selected, 'random_split'].eq('train').all()
        assert chunk.group_id.isin(wanted_groups).equals(selected)
        if selected.any():
            train_parts.append(chunk.loc[selected].copy())
        validation = chunk.random_split.eq('validation')
        if validation.any():
            val_parts.append(chunk.loc[validation].copy())
        # Test records encountered in the source scan are discarded.
    train = pd.concat(train_parts, ignore_index=True)
    val = pd.concat(val_parts, ignore_index=True)
    assert set(train.source_record_id) == wanted_ids
    assert train.source_record_id.is_unique and val.source_record_id.is_unique
    assert set(train.group_id).isdisjoint(val.group_id)
    assert len(train) == 32000 and len(val) == 70040
    assert train.ml_label.nunique() == val.ml_label.nunique() == 24
    assert set(train.ml_label) == set(val.ml_label)
    for frame in [train, val]:
        assert not frame.ml_label.isna().any()
        frame[FEATURES] = frame[FEATURES].astype(np.float64)
        assert np.isfinite(frame[FEATURES].to_numpy()).all()
    return train, val, hashes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir', type=Path, required=True)
    ap.add_argument('--output-root', type=Path, required=True)
    args = ap.parse_args()
    import torch
    import pytorch_lightning as pl
    from pytorch_tabular import TabularModel
    from pytorch_tabular.config import DataConfig, TrainerConfig, OptimizerConfig
    from pytorch_tabular.models import FTTransformerConfig
    from omegaconf import OmegaConf
    if version('pytorch-tabular') != '1.1.1':
        raise ValueError('This pilot targets the inspected PyTorch Tabular 1.1.1 environment.')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    print(f'Python: {sys.executable}; torch: {torch.__version__}; CPU threads: 2', flush=True)
    print('Checking hashes and loading frozen training subset and validation...', flush=True)
    train, val, hashes = load_data(args.data_dir.expanduser().resolve())
    train_df = train[FEATURES + ['ml_label']].copy()
    val_df = val[FEATURES + ['ml_label']].copy()
    val_ids = val.source_record_id.astype(str).to_numpy(dtype=str)
    val_labels = val.ml_label.to_numpy(copy=True)
    reference_scaler = StandardScaler().fit(train_df[FEATURES])
    training_labels = sorted(train.ml_label.unique().tolist())
    del train, val
    out = args.output_root.expanduser().resolve() / datetime.now(timezone.utc).strftime('ft_pilot_%Y%m%dT%H%M%S_%fZ')
    out.mkdir(parents=True, exist_ok=False)
    data_cfg = DataConfig(target=['ml_label'], continuous_cols=FEATURES, categorical_cols=[],
                          normalize_continuous_features=True, continuous_feature_transform=None,
                          num_workers=0, pin_memory=False)
    model_cfg = FTTransformerConfig(task='classification', input_embed_dim=32,
        num_heads=8, num_attn_blocks=3, transformer_head_dim=None,
        attn_dropout=0.1, add_norm_dropout=0.1, ff_dropout=0.1,
        ff_hidden_multiplier=3, transformer_activation='GEGLU',
        attn_feature_importance=False, learning_rate=0.001, seed=42,
        batch_norm_continuous_input=True, loss='CrossEntropyLoss')
    trainer_cfg = TrainerConfig(batch_size=512, max_epochs=2, min_epochs=2,
        accelerator='cpu', devices=1, deterministic=True, seed=42,
        early_stopping=None, checkpoints=None, load_best=False,
        progress_bar='none', precision=32, auto_lr_find=False,
        trainer_kwargs={'enable_model_summary': False, 'num_sanity_val_steps': 0,
                        'default_root_dir': str(out)})
    optimizer_cfg = OptimizerConfig(optimizer='AdamW', optimizer_params={'weight_decay': 0.01})
    tab = TabularModel(data_config=data_cfg, model_config=model_cfg,
                       trainer_config=trainer_cfg, optimizer_config=optimizer_cfg, verbose=False)
    tab.logger = False
    config = dict(purpose='Two-epoch CPU resource pilot; not a selected model',
        features=FEATURES, train_rows=len(train_df), validation_rows=len(val_df),
        epochs=2, seed=42, fit_seed=42, threads=2, interop_threads=1, device='cpu',
        labels=training_labels, input_sha256=hashes, script_sha256=sha(Path(__file__)),
        framework_configuration=OmegaConf.to_container(tab.config, resolve=True),
        versions={name: version(name) for name in ['torch','pytorch-tabular','pytorch-lightning',
                                                  'torchmetrics','numpy','pandas','scikit-learn']},
        python=platform.python_version(), platform=platform.platform(),
        loss='unweighted CrossEntropyLoss', test_evaluated=False,
        scaling='Framework StandardScaler fitted on training float64 features, then float32 tensors; model also uses batch normalization',
        comparison_note='Architecture is a fixed diagnostic configuration, not the archived tuned model.')
    save_json(out / 'pilot_config.json', config)
    history = []

    class Progress(pl.Callback):
        def on_train_epoch_start(self, trainer, module):
            self.epoch_start = time.perf_counter()
            print(f'Epoch {trainer.current_epoch + 1}/2 starts', flush=True)

        def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
            loss = outputs['loss'] if isinstance(outputs, dict) else outputs
            if loss is not None and not torch.isfinite(loss).all():
                raise ValueError('Non-finite training loss')
            if (batch_idx + 1) % 20 == 0:
                print(f'  Training batch {batch_idx + 1}/{trainer.num_training_batches}', flush=True)

        def on_validation_epoch_start(self, trainer, module):
            print('  Validation loss evaluation starts', flush=True)

        def on_train_epoch_end(self, trainer, module):
            row = {'epoch': trainer.current_epoch + 1,
                   'seconds_including_validation': time.perf_counter() - self.epoch_start}
            for key, value in trainer.callback_metrics.items():
                if torch.is_tensor(value) and value.numel() == 1:
                    v = float(value.detach().cpu())
                    if np.isfinite(v):
                        row[str(key)] = v
            history.append(row)
            pd.DataFrame(history).to_csv(out / 'training_history.csv', index=False)
            print(f'Epoch complete: {row}', flush=True)

    print('Train: 32,000; validation: 70,040; six features; batch size: 512; two fixed epochs.', flush=True)
    before = peak_mib()
    t0 = time.perf_counter()
    # Pass the seed to fit explicitly; its default can override configuration seeds.
    tab.fit(train=train_df.copy(), validation=val_df.copy(), seed=42,
            callbacks=[Progress()], handle_oom=False)
    fit_seconds = time.perf_counter() - t0
    after_fit = peak_mib()
    assert len(history) == 2
    encoder = tab.datamodule.label_encoder[0]
    assert encoder.classes_.tolist() == training_labels
    scaler = tab.datamodule.scaler
    np.testing.assert_allclose(scaler.mean_, reference_scaler.mean_, rtol=1e-12, atol=0)
    np.testing.assert_allclose(scaler.scale_, reference_scaler.scale_, rtol=1e-12, atol=0)
    assert int(scaler.n_samples_seen_) == 32000
    save_json(out / 'scaler.json', dict(features=FEATURES, mean=scaler.mean_.tolist(),
        scale=scaler.scale_.tolist(), var=scaler.var_.tolist(), n_samples_seen=32000))
    tab.save_config(str(out))
    tab.save_weights(out / 'pilot_state_dict.pt')
    print('Scoring validation only...', flush=True)
    t0 = time.perf_counter()
    pred_df = tab.predict(val_df[FEATURES].copy(), device='cpu', progress_bar=None)
    predict_seconds = time.perf_counter() - t0
    assert pred_df.index.equals(val_df.index)
    columns = [f'ml_label_{label}_probability' for label in encoder.classes_]
    probs = pred_df[columns].to_numpy()
    assert probs.shape == (70040, 24) and np.isfinite(probs).all()
    assert (probs >= 0).all() and np.allclose(probs.sum(axis=1), 1, atol=1e-5)
    y = encoder.transform(val_labels)
    pred = probs.argmax(axis=1)
    np.testing.assert_array_equal(encoder.inverse_transform(pred), pred_df['ml_label_prediction'].to_numpy())
    np.savez_compressed(out / 'validation_predictions.npz', source_record_id=val_ids,
                         y_true=y, y_pred=pred, probabilities=probs,
                         labels=np.array(training_labels, dtype=str))
    summary = dict(train_rows=32000, validation_rows=70040, feature_count=6, epochs=2,
        trainable_parameters=sum(p.numel() for p in tab.model.parameters() if p.requires_grad),
        fit_seconds=fit_seconds, epoch_history=history,
        validation_prediction_batch_seconds=predict_seconds,
        validation_amortized_microseconds_per_row=predict_seconds/70040*1e6,
        validation_weighted_f1=float(f1_score(y, pred, average='weighted', zero_division=0)),
        validation_macro_f1_all24=float(f1_score(y, pred, labels=np.arange(24), average='macro', zero_division=0)),
        validation_accuracy=float(accuracy_score(y, pred)),
        process_peak_rss_before_fit_mib=before, process_peak_rss_after_fit_mib=after_fit,
        process_peak_rss_mib=peak_mib(),
        serialized_weights_mib=(out / 'pilot_state_dict.pt').stat().st_size/1024**2,
        test_evaluated=False, scaler_training_only_verified=True, output_directory=str(out),
        elapsed_seconds_before_summary=time.perf_counter()-started,
        interpretation='Fixed two-epoch 32k resource probe, not a final score, convergence assessment, or full-workload estimate.',
        timing_note='Fit includes data/model preparation and full validation each epoch. Prediction includes framework preprocessing and minibatch inference, without separate warm-up; not single-event latency.',
        memory_note='Fresh-process peak RSS includes imported libraries, data, optimizer and model; not model-only or whole-system memory.',
        size_note='State dictionary only; preprocessing/configuration saved separately, not a deployment-size comparison.')
    save_json(out / 'pilot_summary.json', summary)
    print('\nPILOT COMPLETE\n' + json.dumps(summary, indent=2), flush=True)
    print('\nShare pilot_summary.json and pilot_config.json. No search or test evaluation was performed.', flush=True)


if __name__ == '__main__':
    main()
