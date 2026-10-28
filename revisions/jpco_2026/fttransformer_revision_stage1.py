"""JPCO FT-Transformer stage 1: training-only CV and outer validation.

12 configurations x 3 folds x 4 conditions = 144 CV fits, then four full
training fits, all with 20 fixed epochs. Resume completed folds by rerunning
the identical command; do not change the script, inputs, or environment.
Uses PyTorch Tabular 1.1.1, CPU, two threads, batch 512, explicit fit seeds.
No test evaluation, historical hyperparameter reuse, or outer-validation
selection. This bounded revision search does not reproduce historical Optuna.

Place beside the unchanged xgboost_revision_stage1.py. That version-checked
helper supplies the same grouped/random and forward temporal folds as XGB/MLP.
Temporal folds have no embargo. Fits use unweighted CE, training-only float64
StandardScaler followed by float32 tensors and model batch normalization.

PyTorch Tabular fits a label encoder to each training fold. If a class is
absent, its global probability is zero when mapping fold predictions to the
fixed 24-class evaluation order; no artificial training rows are introduced.
This differs from MLP's fixed 24-output CV model and is recorded in protocol.
Full outer training contains all 24 classes.

To avoid the API's automatic inner split, fit receives a small training-only
placeholder as validation, with limit_val_batches=0. No validation batches,
early stopping or best-checkpoint reload occur; holdouts are scored once after
20 epochs. The placeholder adds no rows to the training dataset.
Timing and cumulative process RSS are diagnostics, not single-event latency
or model-only memory. State dictionary size excludes preprocessing.
"""
import argparse
import gc
import hashlib
import importlib.util
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import random
import time

for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[name] = '2'

import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Reuse exactly the audited fold construction; the helper does not run main on import.
HELPER_SHA = '7c06aa5cffb280dea07ec88ce4c140e9d6a98db6574996b60a377d0946c78b7b'
helper_path = Path(__file__).resolve().with_name('xgboost_revision_stage1.py')
if not helper_path.is_file():
    raise RuntimeError('Put this script beside your unchanged xgboost_revision_stage1.py.')
if hashlib.sha256(helper_path.read_bytes()).hexdigest() != HELPER_SHA:
    raise RuntimeError('XGBoost helper differs from the audited version. Share this error.')
spec = importlib.util.spec_from_file_location('frozen_split_helpers', helper_path)
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)

EXPECTED_INPUTS = {
    'manifest.json': 'f1eedfaa282d6ae90e1567d271f9baff06c62ba233aff5890ae5dad85cfe9605',
    'o3_primary.csv.gz': '9d7d7d217563ccb121eba466fc442d83d7f25495e6e5ba40ea9a0962114829c6',
    'split_summary.csv': '61de488a44179ca86205ffcb45e98baa9eb2d9a7b52a1ab13cf2845368f88d01',
    'tuning_random_split.csv.gz': '8afbc1e03efa547ae4d51dcd3982308ebaa67f018f2fc225b59daaa55f153342',
    'tuning_chronological_split.csv.gz': 'bd4b6e9c563f0db8d0d3b0c15677ccf8290d602a589d2cf8ea482d7baf4a2f11',
}
GRID = [dict(num_attn_blocks=blocks, dropout=dropout, learning_rate=lr)
        for blocks in [2, 3]
        for dropout in [0.0, 0.2]
        for lr in [0.0003, 0.001, 0.003]]
EPOCHS = 20


def fit_model(torch, X, y, candidate, seed, log_prefix, features):
    from pytorch_tabular import TabularModel
    from pytorch_tabular.config import DataConfig, TrainerConfig, OptimizerConfig
    from pytorch_tabular.models import FTTransformerConfig
    import pytorch_lightning as pl
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    params = GRID[candidate]
    started = time.perf_counter()
    assert X.dtype == np.float64 and np.isfinite(X).all()
    df = pd.DataFrame(X, columns=features)
    df['ml_label'] = y.astype(np.int64)
    reference = StandardScaler().fit(X)
    data = DataConfig(target=['ml_label'], continuous_cols=features, categorical_cols=[],
                      normalize_continuous_features=True, continuous_feature_transform=None,
                      num_workers=0, pin_memory=False)
    architecture = FTTransformerConfig(task='classification', input_embed_dim=32, num_heads=8,
        num_attn_blocks=params['num_attn_blocks'], transformer_head_dim=None,
        attn_dropout=params['dropout'], add_norm_dropout=params['dropout'],
        ff_dropout=params['dropout'], ff_hidden_multiplier=3, transformer_activation='GEGLU',
        attn_feature_importance=False, learning_rate=params['learning_rate'], seed=seed,
        batch_norm_continuous_input=True, loss='CrossEntropyLoss')
    training = TrainerConfig(batch_size=512, max_epochs=EPOCHS, min_epochs=EPOCHS,
        accelerator='cpu', devices=1, deterministic=True, seed=seed,
        early_stopping=None, checkpoints=None, load_best=False,
        progress_bar='none', precision=32, auto_lr_find=False,
        trainer_kwargs={'enable_model_summary': False, 'num_sanity_val_steps': 0,
                        'limit_val_batches': 0})
    optimizer = OptimizerConfig(optimizer='AdamW', optimizer_params={'weight_decay': 0.01})
    tab = TabularModel(data_config=data, model_config=architecture, trainer_config=training,
                       optimizer_config=optimizer, verbose=False)
    tab.logger = False
    history = []

    class Progress(pl.Callback):
        def on_train_epoch_start(self, trainer, module):
            self.t0 = time.perf_counter()
            self.total_loss = 0.0
            self.n = 0
        def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
            loss = outputs['loss'] if isinstance(outputs, dict) else outputs
            if loss is None or not torch.isfinite(loss).all():
                raise ValueError(f'Non-finite/missing loss: {log_prefix}')
            size = batch['continuous'].shape[0]
            self.total_loss += float(loss.detach().cpu()) * size
            self.n += size
        def on_train_epoch_end(self, trainer, module):
            row = dict(epoch=trainer.current_epoch+1, loss=self.total_loss/self.n,
                       seconds=time.perf_counter()-self.t0, rows_seen=self.n)
            history.append(row)
            print(f"{log_prefix}: epoch {row['epoch']}/{EPOCHS}; loss={row['loss']:.5f}; {row['seconds']:.1f}s", flush=True)

    # Explicit placeholder prevents automatic 20% holdout; validation loop is disabled.
    tab.fit(train=df, validation=df.iloc[:min(512, len(df))].copy(), seed=seed,
            callbacks=[Progress()], handle_oom=False)
    seconds = time.perf_counter()-started
    assert len(history) == EPOCHS
    assert tab.trainer.limit_val_batches == 0
    assert len(tab.datamodule.train_dataset) == len(y)
    scaler = tab.datamodule.scaler
    np.testing.assert_allclose(scaler.mean_, reference.mean_, rtol=1e-12, atol=0)
    np.testing.assert_allclose(scaler.scale_, reference.scale_, rtol=1e-12, atol=0)
    assert int(scaler.n_samples_seen_) == len(y)
    np.testing.assert_array_equal(tab.datamodule.label_encoder[0].classes_, np.unique(y))
    return tab, scaler, history, seconds


def predict(torch, model, scaler, X):
    started = time.perf_counter()
    features = list(model.config.continuous_cols)
    assert np.isfinite(X).all()
    frame = pd.DataFrame(X, columns=features)
    result = model.predict(frame, device='cpu', progress_bar=None)
    assert result.index.equals(frame.index)
    local_labels = np.asarray(model.datamodule.label_encoder[0].classes_, dtype=np.int64)
    assert ((local_labels >= 0) & (local_labels < 24)).all()
    columns = [f'ml_label_{label}_probability' for label in local_labels]
    probabilities = np.zeros((len(X), 24), dtype=np.float32)
    probabilities[:, local_labels] = result[columns].to_numpy()
    assert np.isfinite(probabilities).all() and (probabilities >= 0).all()
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)
    np.testing.assert_array_equal(probabilities.argmax(axis=1), result['ml_label_prediction'].to_numpy())
    return probabilities, time.perf_counter()-started


def load_data(base, split, features, counts):
    members = pd.read_csv(base / f'tuning_{split}.csv.gz')
    ids, groups = set(members.source_record_id), set(members.group_id)
    columns = list(dict.fromkeys(['source_record_id', 'group_id', 'peak_gps_ns', 'ml_label', split] + features))
    tr_parts, va_parts = [], []
    for chunk in pd.read_csv(base / 'o3_primary.csv.gz', usecols=columns, chunksize=50000,
                             float_precision='round_trip'):
        assert chunk.loc[chunk.source_record_id.isin(ids), split].eq('train').all()
        tr_parts.append(chunk.loc[chunk[split].eq('train')].copy())
        va_parts.append(chunk.loc[chunk[split].eq('validation')].copy())
    train = pd.concat(tr_parts, ignore_index=True)
    val = pd.concat(va_parts, ignore_index=True)
    for role, frame in [('train', train), ('validation', val)]:
        n = counts.loc[(counts.configuration == split) & (counts.partition == role), 'rows'].item()
        assert len(frame) == n and frame.source_record_id.is_unique
    assert set(train.group_id).isdisjoint(val.group_id)
    selected = train.source_record_id.isin(ids)
    assert selected.equals(train.group_id.isin(groups))
    tune = train.loc[selected].reset_index(drop=True)
    assert set(tune.source_record_id) == ids
    encoder = LabelEncoder().fit(train.ml_label)
    assert len(encoder.classes_) == 24 and set(val.ml_label) <= set(encoder.classes_)
    if split == 'chronological_split':
        assert train.peak_gps_ns.max() < val.peak_gps_ns.min()
    return train, val, tune, encoder


def main():
    import torch
    if version('pytorch-tabular') != '1.1.1':
        raise RuntimeError('Expected inspected PyTorch Tabular 1.1.1 environment.')
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    base = args.data_dir.expanduser().resolve()
    out = args.output_dir.expanduser().resolve()
    print('Verifying frozen input hashes...', flush=True)
    for name, expected in EXPECTED_INPUTS.items():
        assert core.sha(base/name) == expected, f'Input mismatch: {name}'
    manifest = json.loads((base/'manifest.json').read_text())
    protocol = dict(stage='training-only FT-Transformer CV and outer validation; test not evaluated',
                    version='1.0', script_sha256=core.sha(Path(__file__)), helper_sha256=HELPER_SHA,
                    inputs=EXPECTED_INPUTS, candidate_grid=GRID, epochs=EPOCHS,
                    activation='GEGLU', optimizer='AdamW', weight_decay=0.01, batch_size=512,
                    input_embed_dim=32, num_heads=8, ff_hidden_multiplier=3,
                    batch_norm_continuous_input=True, attn_feature_importance=False,
                    fold_missing_class_handling='fit local label encoder; map probabilities to global 24 with zero for absent classes',
                    validation_during_fit='disabled; training-only placeholder prevents automatic inner split',
                    objective='unweighted cross entropy', conditions=core.CONDITIONS,
                    features=manifest['feature_sets'], folds=3,
                    cv_random='StratifiedGroupKFold, seed42; same helper and data order as XGBoost',
                    cv_temporal='3 expanding timestamp-block folds; same helper; no embargo',
                    selection='highest mean fold weighted F1; ties choose earlier candidate',
                    cv_training_seeds=[42,43,44], full_training_seed=42,
                    numeric_representation='float64 training-fitted StandardScaler before float32 tensors',
                    historical_protocol_reproduced=False, early_stopping=False,
                    outer_validation_used_for_selection=False, test_evaluated=False,
                    threads=2, interop_threads=1, device='cpu',
                    versions=dict(python=platform.python_version(), torch=str(torch.__version__),
                                  numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__,
                                  pytorch_tabular=version('pytorch-tabular'), pytorch_lightning=version('pytorch-lightning'),
                                  torchmetrics=version('torchmetrics')),
                    platform=platform.platform(), architecture=platform.machine())
    protocol = json.loads(json.dumps(protocol))
    out.mkdir(parents=True, exist_ok=True)
    lock = out/'protocol.json'
    if lock.exists():
        assert json.loads(lock.read_text()) == protocol, 'Protocol/environment changed. Do not mix runs.'
    else:
        assert not any(out.iterdir()), 'Output directory contains files without a protocol.'
        core.save_json(lock, protocol)
    protocol_hash = core.sha(lock)
    counts = pd.read_csv(base/'split_summary.csv')
    summaries = []
    for split, feature_set in core.CONDITIONS:
        tag = f'{split}__{feature_set}'
        folder = out/tag
        folder.mkdir(exist_ok=True)
        print('\nCONDITION: '+tag, flush=True)
        final = folder/'validation_result.json'
        if final.exists():
            result = json.loads(final.read_text())
            assert result['protocol_sha256'] == protocol_hash
            for name, expected in result['artifact_sha256'].items():
                assert core.sha(folder/name) == expected, f'Completed artifact changed: {name}'
            summaries.append(result)
            print('Using verified completed condition.', flush=True)
            continue
        features = manifest['feature_sets'][feature_set]
        train, val, tune, encoder = load_data(base, split, features, counts)
        X = tune[features].to_numpy(dtype=np.float64)
        y = encoder.transform(tune.ml_label)
        assert np.isfinite(X).all()
        folds = core.make_folds(tune, y, split)
        records, diagnostics = [], []
        for k, (i,j) in enumerate(folds):
            missing = sorted(set(range(24)) - set(y[i]))
            diagnostics.append(dict(fold=k, train_rows=len(i), validation_rows=len(j),
                                    training_missing_classes=encoder.inverse_transform(missing).tolist() if missing else [],
                                    validation_classes=int(len(np.unique(y[j]))),
                                    validation_rows_with_unseen_training_label=int((~np.isin(y[j],y[i])).sum())))
            for role, ix in [('train',i),('validation',j)]:
                records.append(tune.iloc[ix][['source_record_id','group_id']].assign(fold=k, role=role))
        membership = pd.concat(records, ignore_index=True)
        membership_path = folder/'cv_membership.csv.gz'
        if membership_path.exists():
            pd.testing.assert_frame_equal(pd.read_csv(membership_path), membership, check_dtype=False)
        else:
            membership.to_csv(membership_path, index=False, compression='gzip')
        core.save_json(folder/'fold_diagnostics.json', diagnostics)
        del records, membership
        trials = []
        for candidate in range(len(GRID)):
            fold_results = []
            for k, (i,j) in enumerate(folds):
                checkpoint = folder/f'candidate_{candidate:02d}_fold_{k}.json'
                if checkpoint.exists():
                    result = json.loads(checkpoint.read_text())
                    assert result['protocol_sha256'] == protocol_hash
                    assert result['candidate'] == candidate and result['fold'] == k
                else:
                    prefix = f'  candidate {candidate+1}/12 fold {k+1}/3'
                    model, scaler, history, seconds = fit_model(torch, X[i], y[i], candidate, 42+k, prefix, features)
                    prob, pred_seconds = predict(torch, model, scaler, X[j])
                    result = dict(candidate=candidate, fold=k, seed=42+k, fit_seconds=seconds,
                                  protocol_sha256=protocol_hash, history=history,
                                  prediction_seconds=pred_seconds, **core.scores(y[j], prob.argmax(axis=1)))
                    core.save_json(checkpoint, result)
                    del model, scaler, history, prob
                    gc.collect()
                fold_results.append(result)
                print(f"  candidate {candidate+1}/12 fold {k+1}/3: weighted F1={result['weighted_f1']:.5f}; {result['fit_seconds']:.1f}s", flush=True)
            trials.append(dict(candidate=candidate,
                               cv_mean_weighted_f1=float(np.mean([r['weighted_f1'] for r in fold_results])),
                               cv_fit_seconds=sum(r['fit_seconds'] for r in fold_results)))
            pd.DataFrame(trials).to_csv(folder/'cv_results.csv', index=False)
        best = max(trials, key=lambda r:r['cv_mean_weighted_f1'])
        choice = best['candidate']
        core.save_json(folder/'selected_configuration.json', dict(**best, params=GRID[choice],
                       epochs=EPOCHS, activation='GEGLU', weight_decay=0.01, batch_size=512,
                       features=features, labels=encoder.classes_.tolist(), full_training_seed=42,
                       protocol_sha256=protocol_hash, selection_used_outer_validation=False))
        Xt = train[features].to_numpy(dtype=np.float64)
        Xv = val[features].to_numpy(dtype=np.float64)
        yt, yv = encoder.transform(train.ml_label), encoder.transform(val.ml_label)
        val_ids = val.source_record_id.astype(str).to_numpy(dtype=str)
        assert np.isfinite(Xt).all() and np.isfinite(Xv).all()
        del train, val, tune, X, y, folds
        gc.collect()
        print(f'Selected candidate {choice+1}; fitting all {len(yt):,} training rows.', flush=True)
        model, scaler, history, seconds = fit_model(torch, Xt, yt, choice, 42, '  full training', features)
        prob, pred_seconds = predict(torch, model, scaler, Xv)
        pred = prob.argmax(axis=1)
        model.save_weights(folder/'selected_state_dict.pt')
        model.save_config(str(folder))
        assert model.datamodule.label_encoder[0].classes_.tolist() == list(range(24))
        core.save_json(folder/'scaler.json', dict(features=features, mean=scaler.mean_.tolist(),
                       scale=scaler.scale_.tolist(), var=scaler.var_.tolist(),
                       n_samples_seen=int(scaler.n_samples_seen_)))
        pd.DataFrame(history).to_csv(folder/'training_history.csv', index=False)
        np.savez_compressed(folder/'validation_predictions.npz', source_record_id=val_ids,
                            y_true=yv, y_pred=pred, probabilities=prob)
        artifact_names = ['selected_state_dict.pt','scaler.json','selected_configuration.json',
                          'validation_predictions.npz','training_history.csv','cv_membership.csv.gz','config.yml']
        result = dict(condition=tag, split=split, feature_set=feature_set, candidate=choice,
                      cv_mean_weighted_f1=best['cv_mean_weighted_f1'], train_rows=len(yt), validation_rows=len(yv),
                      validation_supported_classes=int(len(np.unique(yv))), fit_seconds=seconds,
                      validation_prediction_batch_seconds=pred_seconds,
                      trainable_parameters=sum(p.numel() for p in model.model.parameters() if p.requires_grad),
                      weights_mib=(folder/'selected_state_dict.pt').stat().st_size/1024**2,
                      process_lifetime_peak_rss_mib=core.peak_mib(), test_evaluated=False,
                      protocol_sha256=protocol_hash,
                      artifact_sha256={name:core.sha(folder/name) for name in artifact_names},
                      **{'validation_'+key:value for key,value in core.scores(yv,pred).items()})
        core.save_json(final, result)
        summaries.append(result)
        pd.DataFrame(summaries).drop(columns=['artifact_sha256']).to_csv(out/'validation_summary.csv', index=False)
        del model, scaler, history, prob, pred, Xt, Xv, yt, yv
        gc.collect()
        print(json.dumps(result, indent=2), flush=True)
    summary = pd.DataFrame(summaries).drop(columns=['artifact_sha256'])
    summary.to_csv(out/'validation_summary.csv', index=False)
    print('\nFT-TRANSFORMER STAGE 1 COMPLETE; no test evaluation.', flush=True)
    print(summary[['condition','cv_mean_weighted_f1','validation_weighted_f1','validation_macro_f1_all24']].to_string(index=False), flush=True)
    print('\nShare protocol.json and validation_summary.csv from this FT-Transformer output folder. Keep every checkpoint.', flush=True)


if __name__ == '__main__':
    main()
