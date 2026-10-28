"""Frozen three-seed MLP test evaluation for the reviewed stage-1 run.

Requires unchanged mlp_revision_stage1.py and xgboost_revision_stage1.py beside
this file. Reuse stage-1 seed42, train seeds43/44 on outer TRAIN only. No search,
early stopping, best-test-seed selection, or train+validation refit.

All 12 configurations/fold results are checked before opening test partitions.
Twenty epochs, architecture and training settings remain frozen. Summary SD is
sample SD (ddof=1) across three training seeds, not a confidence interval.
Paired timestamp differences include independently selected hyperparameters;
cross-protocol scores involve different event populations and class mixtures.

Training-fitted float64 StandardScaler precedes float32 conversion, unlike the
raw-float32 XGBoost representation. The MLP loss is unweighted; XGBoost used
inverse-frequency weights. These are model-specific training procedures, not
identical numerical pipelines. Timings/peak RSS remain scheduling diagnostics.
Saved state-dict size excludes preprocessing and is not deployment footprint.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil

for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[name] = '2'

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler

SEEDS = [42,43,44]
EXPECTED_PROTOCOL_SHA = '2cd30e32ca65a3f6f33633da2ec873516ff804b1faf41f9b996a3f206129a801'
EXPECTED_STAGE1_SHA = '1aed088a3fa04eaed98c5fa081328b606a3085b43c2b3bd71a44bf1f14d18b93'
EXPECTED_CHOICES = {
    'random_split__original_nine': 8,
    'random_split__timestamp_free_six': 11,
    'chronological_split__original_nine': 11,
    'chronological_split__timestamp_free_six': 8,
}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text())


def validate_cv(folder, choice):
    means = []
    for candidate in range(12):
        scores = []
        for fold in range(3):
            r = read(folder/f'candidate_{candidate:02d}_fold_{fold}.json')
            assert r['protocol_sha256'] == EXPECTED_PROTOCOL_SHA
            assert r['candidate'] == candidate and r['fold'] == fold and r['seed'] == 42+fold
            value = float(r['weighted_f1'])
            assert np.isfinite(value) and 0 <= value <= 1
            scores.append(value)
        means.append(float(np.mean(scores)))
    assert int(np.argmax(means)) == choice, 'CV checkpoints disagree with frozen selection.'
    return means[choice]


def load_train_test(base, split, features, counts):
    columns = list(dict.fromkeys(['source_record_id','group_id','peak_gps_ns','ml_label',split]+features))
    a,b = [],[]
    for chunk in pd.read_csv(base/'o3_primary.csv.gz',usecols=columns,float_precision='round_trip',chunksize=50000):
        a.append(chunk.loc[chunk[split].eq('train')].copy())
        b.append(chunk.loc[chunk[split].eq('test')].copy())
    train,test = pd.concat(a,ignore_index=True),pd.concat(b,ignore_index=True)
    for partition,frame in [('train',train),('test',test)]:
        n = counts.loc[(counts.configuration==split)&(counts.partition==partition),'rows'].item()
        assert len(frame)==n and frame.source_record_id.is_unique
    assert set(train.group_id).isdisjoint(test.group_id)
    if split=='chronological_split':
        assert train.peak_gps_ns.max()<test.peak_gps_ns.min()
    return train,test


def build_model(torch, features, params):
    layers=[]
    previous=len(features)
    for width in params['hidden_dims']:
        layers.extend([torch.nn.Linear(previous,width),torch.nn.Mish(),torch.nn.Dropout(params['dropout'])])
        previous=width
    layers.append(torch.nn.Linear(previous,24))
    return torch.nn.Sequential(*layers).cpu()


def restore_scaler(value, X, features):
    assert value['features']==features
    # Refit on TRAIN solely to verify saved preprocessing, never on test records.
    verified=StandardScaler().fit(X)
    for key,attribute in [('mean','mean_'),('scale','scale_'),('var','var_')]:
        np.testing.assert_array_equal(getattr(verified,attribute),np.asarray(value[key],dtype=np.float64))
    assert int(verified.n_samples_seen_)==value['n_samples_seen']
    return verified


def main():
    import torch
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir',type=Path,required=True)
    ap.add_argument('--stage1-dir',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,required=True)
    args=ap.parse_args()
    base=args.data_dir.expanduser().resolve()
    previous=args.stage1_dir.expanduser().resolve()
    out=args.output_dir.expanduser().resolve()
    assert sha(previous/'protocol.json')==EXPECTED_PROTOCOL_SHA,'Stage-1 protocol changed.'
    protocol=read(previous/'protocol.json')
    source=Path(__file__).resolve().with_name('mlp_revision_stage1.py')
    assert sha(source)==EXPECTED_STAGE1_SHA==protocol['script_sha256'],'Restore unchanged MLP stage-1 script.'
    spec=importlib.util.spec_from_file_location('frozen_mlp_stage1',source)
    stage1=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stage1)
    core=stage1.core
    versions=dict(python=platform.python_version(),torch=str(torch.__version__),
                  numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__)
    assert versions==protocol['versions'],'Use the same env1 package versions as stage 1.'
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    print('Verifying inputs, saved models, scalers and all CV selections...',flush=True)
    for name,expected in protocol['inputs'].items():
        assert sha(base/name)==expected,f'Input changed: {name}'
    dataset=read(base/'manifest.json')
    choices,preflight={},{}
    for split,feature_set in core.CONDITIONS:
        tag=f'{split}__{feature_set}'
        folder=previous/tag
        selected=read(folder/'selected_configuration.json')
        result=read(folder/'validation_result.json')
        choice=EXPECTED_CHOICES[tag]
        assert result['protocol_sha256']==selected['protocol_sha256']==EXPECTED_PROTOCOL_SHA
        assert not result['test_evaluated'] and not selected['selection_used_outer_validation']
        assert result['candidate']==selected['candidate']==choice
        assert selected['params']==stage1.GRID[choice]
        assert selected['features']==dataset['feature_sets'][feature_set]
        assert selected['epochs']==20 and selected['activation']=='Mish'
        assert selected['weight_decay']==0.01 and selected['batch_size']==1024
        assert selected['full_training_seed']==42
        cv=validate_cv(folder,choice)
        assert np.isclose(cv,selected['cv_mean_weighted_f1'],rtol=0,atol=1e-12)
        assert np.isclose(cv,result['cv_mean_weighted_f1'],rtol=0,atol=1e-12)
        for name,expected in result['artifact_sha256'].items():
            assert sha(folder/name)==expected,f'Stage-1 artifact changed: {tag}/{name}'
        choices[tag]=selected
        preflight[tag]=dict(candidate=choice,params=selected['params'],features=selected['features'],
                            seed42_model_sha256=sha(folder/'selected_state_dict.pt'),
                            scaler_sha256=sha(folder/'scaler.json'),
                            selected_configuration_sha256=sha(folder/'selected_configuration.json'))
    frozen=dict(stage='MLP fixed three-seed test evaluation',seeds=SEEDS,
                script_sha256=sha(Path(__file__)),stage1_protocol_sha256=EXPECTED_PROTOCOL_SHA,
                inputs=protocol['inputs'],selections=preflight,versions=versions,
                epochs=20,training_scope='outer training only; no validation refit',
                test_used_for_selection=False,
                metrics=['weighted_f1','macro_f1_all24','macro_f1_supported','accuracy'],
                aggregation='arithmetic mean and sample SD ddof=1; not a confidence interval')
    out.mkdir(parents=True,exist_ok=True)
    lock=out/'frozen_test_protocol.json'
    if lock.exists():
        assert read(lock)==frozen,'Existing test protocol differs. Do not mix runs.'
    else:
        assert not any(out.iterdir()),'Output folder contains files without a frozen protocol.'
        core.save_json(lock,frozen)
    lock_hash=sha(lock)
    counts=pd.read_csv(base/'split_summary.csv')
    all_results,all_class_rows=[],[]
    for split,feature_set in core.CONDITIONS:
        tag=f'{split}__{feature_set}'
        selected=choices[tag]
        features=selected['features']
        folder=out/tag
        folder.mkdir(exist_ok=True)
        print('\nTEST CONDITION: '+tag,flush=True)
        train,test=load_train_test(base,split,features,counts)
        encoder=LabelEncoder().fit(train.ml_label)
        assert encoder.classes_.tolist()==selected['labels'] and len(encoder.classes_)==24
        assert set(test.ml_label)<=set(train.ml_label)
        Xt,Xe=train[features].to_numpy(dtype=np.float64),test[features].to_numpy(dtype=np.float64)
        yt,ye=encoder.transform(train.ml_label),encoder.transform(test.ml_label)
        assert np.isfinite(Xt).all() and np.isfinite(Xe).all()
        metadata=test[['source_record_id','group_id','peak_gps_ns','ml_label']].reset_index(drop=True)
        meta_path=folder/'test_records.csv.gz'
        if meta_path.exists():
            pd.testing.assert_frame_equal(pd.read_csv(meta_path),metadata,check_dtype=False)
        else:
            metadata.to_csv(meta_path,index=False,compression='gzip')
        core.save_json(folder/'label_mapping.json',encoder.classes_.tolist())
        scaler_value=read(previous/tag/'scaler.json')
        verified_scaler=restore_scaler(scaler_value,Xt,features)
        del train,test
        gc.collect()
        for seed in SEEDS:
            result_path=folder/f'seed_{seed}_test_metrics.json'
            pred_path=folder/f'seed_{seed}_predictions.npz'
            class_path=folder/f'seed_{seed}_class_report.json'
            model_path=folder/f'seed_{seed}_state_dict.pt'
            scaler_path=folder/f'seed_{seed}_scaler.json'
            history_path=folder/f'seed_{seed}_training_history.csv'
            if result_path.exists():
                result=read(result_path)
                assert result['frozen_test_protocol_sha256']==lock_hash
                assert result['seed']==seed and result['condition']==tag
                for name,expected in result['artifact_sha256'].items():
                    assert sha(folder/name)==expected,f'Completed artifact changed: {name}'
                with np.load(pred_path,allow_pickle=False) as saved:
                    assert np.array_equal(saved['true_label'],ye)
                    pred=saved['prediction']
                for key,value in core.scores(ye,pred).items():
                    assert np.isclose(result['test_'+key],value,rtol=0,atol=1e-12)
                report=read(class_path)
                print(f'  seed {seed}: verified completed checkpoint',flush=True)
            else:
                if seed==42:
                    print('  seed42: reusing reviewed stage-1 model',flush=True)
                    model=build_model(torch,features,selected['params'])
                    model.load_state_dict(torch.load(previous/tag/'selected_state_dict.pt',map_location='cpu',weights_only=True))
                    scaler=verified_scaler
                    fit_seconds=None
                    shutil.copy2(previous/tag/'selected_state_dict.pt',model_path)
                    shutil.copy2(previous/tag/'training_history.csv',history_path)
                else:
                    print(f'  seed{seed}: training {len(yt):,} rows for 20 epochs',flush=True)
                    model,scaler,history,fit_seconds=stage1.fit_model(torch,Xt,yt,selected['candidate'],seed,f'  seed{seed}')
                    np.testing.assert_array_equal(scaler.mean_,verified_scaler.mean_)
                    np.testing.assert_array_equal(scaler.scale_,verified_scaler.scale_)
                    torch.save(model.state_dict(),model_path)
                    pd.DataFrame(history).to_csv(history_path,index=False)
                    del history
                core.save_json(scaler_path,scaler_value)
                prob,pred_seconds=stage1.predict(torch,model,scaler,Xe)
                pred=prob.argmax(axis=1).astype(np.int16)
                np.savez_compressed(pred_path,prediction=pred,true_label=ye.astype(np.int16))
                report=classification_report(ye,pred,labels=np.arange(24),target_names=encoder.classes_.tolist(),
                                             zero_division=0,output_dict=True)
                core.save_json(class_path,report)
                names=[model_path.name,scaler_path.name,history_path.name,pred_path.name,class_path.name,
                       'test_records.csv.gz','label_mapping.json']
                result=dict(condition=tag,split=split,feature_set=feature_set,seed=seed,
                            candidate=selected['candidate'],train_rows=len(yt),test_rows=len(ye),
                            test_supported_classes=int(len(np.unique(ye))),reused_stage1_model=seed==42,
                            fit_seconds=fit_seconds,test_prediction_batch_seconds=pred_seconds,
                            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                            weights_mib=model_path.stat().st_size/1024**2,model_sha256=sha(model_path),
                            process_lifetime_peak_rss_mib=core.peak_mib(),test_evaluated=True,
                            frozen_test_protocol_sha256=lock_hash,
                            artifact_sha256={name:sha(folder/name) for name in names},
                            **{'test_'+key:value for key,value in core.scores(ye,pred).items()})
                core.save_json(result_path,result)
                del model,scaler,prob
                gc.collect()
                print(f"  seed {seed}: test weighted F1={result['test_weighted_f1']:.6f}",flush=True)
            all_results.append({key:value for key,value in result.items() if key!='artifact_sha256'})
            for label in encoder.classes_:
                r=report[label]
                all_class_rows.append(dict(condition=tag,seed=seed,label=label,support=int(r['support']),
                                            precision=r['precision'],recall=r['recall'],f1=r['f1-score']))
            pd.DataFrame(all_results).to_csv(out/'test_per_seed.csv',index=False)
        del Xt,Xe,yt,ye,metadata,pred,verified_scaler
        gc.collect()
    raw=pd.DataFrame(all_results)
    rows=[]
    for tag,g in raw.groupby('condition',sort=False):
        assert len(g)==3 and set(g.seed)==set(SEEDS)
        r=dict(condition=tag,train_rows=int(g.train_rows.iloc[0]),test_rows=int(g.test_rows.iloc[0]),
               supported_classes=int(g.test_supported_classes.iloc[0]),seeds=len(g))
        for metric in frozen['metrics']:
            r[metric+'_mean']=float(g['test_'+metric].mean())
            r[metric+'_sd']=float(g['test_'+metric].std(ddof=1))
        rows.append(r)
    summary=pd.DataFrame(rows)
    summary.to_csv(out/'test_summary.csv',index=False)
    pd.DataFrame(all_class_rows).to_csv(out/'test_class_metrics.csv',index=False)
    differences=[]
    for split in ['random_split','chronological_split']:
        a=raw[(raw.split==split)&(raw.feature_set=='original_nine')].set_index('seed')
        b=raw[(raw.split==split)&(raw.feature_set=='timestamp_free_six')].set_index('seed')
        left=pd.read_csv(out/f'{split}__original_nine'/'test_records.csv.gz',usecols=['source_record_id','ml_label'])
        right=pd.read_csv(out/f'{split}__timestamp_free_six'/'test_records.csv.gz',usecols=['source_record_id','ml_label'])
        assert left.equals(right)
        diff=(a.test_weighted_f1-b.test_weighted_f1)*100
        differences.append(dict(split=split,nine_minus_six_weighted_f1_pp_mean=float(diff.mean()),
                                 nine_minus_six_weighted_f1_pp_sd=float(diff.std(ddof=1))))
    pd.DataFrame(differences).to_csv(out/'timestamp_differences.csv',index=False)
    print('\nMLP STAGE 2 COMPLETE. Keep configurations fixed; do not retune from test scores.',flush=True)
    print(summary.to_string(index=False),flush=True)
    print('\nShare test_summary.csv, test_per_seed.csv, test_class_metrics.csv and timestamp_differences.csv from this MLP output folder.',flush=True)


if __name__=='__main__':
    main()
