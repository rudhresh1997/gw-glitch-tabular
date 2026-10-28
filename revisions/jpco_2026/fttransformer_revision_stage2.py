"""FT-Transformer frozen three-seed test evaluation for the reviewed stage-1 run.

Place beside unchanged fttransformer_revision_stage1.py and
xgboost_revision_stage1.py. Verify all CV choices, hashes and settings before
loading any test partition. Reuse stage-1 seed42; train seeds43/44 on outer
training only for the same 20 epochs. No search, best-test-seed selection,
validation refit, early stopping, or change to hyperparameters.

Restore seed42 from a state dictionary with weights_only=True, rebuilding
preprocessing from training data only and verifying it against saved values.
Reproduce saved validation predictions before evaluating its test predictions.
Completed seed evaluations resume after artifact verification.

Sample SD across three training seeds (ddof=1) is not a confidence interval.
Timestamp comparisons include independently CV-selected configurations.
Random and chronological test sets contain different events/class mixtures.
Unweighted CE and float64 StandardScaler followed by float32 tensors match
FT stage1, not the inverse-frequency loss/raw float32 XGBoost procedure.
Timing and cumulative peak RSS are scheduling diagnostics; state-dictionary
size excludes preprocessing. Do not label these as deployment benchmarks.
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
EXPECTED_PROTOCOL_SHA = 'c2df0ad2ad895d722b803392616823ce22cc6f60a199e777346a3fd53ccc0dd4'
EXPECTED_STAGE1_SHA = 'bf5dd746c97db114a3054416adb46e4d7df79a056d6ea74adfaf4931bcf07af3'
EXPECTED_CHOICES = {
    'random_split__original_nine': 10,
    'random_split__timestamp_free_six': 7,
    'chronological_split__original_nine': 2,
    'chronological_split__timestamp_free_six': 2,
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


def restore_scaler(value, X, features):
    assert value['features']==features
    verified=StandardScaler().fit(pd.DataFrame(X,columns=features))
    for key,attribute in [('mean','mean_'),('scale','scale_'),('var','var_')]:
        np.testing.assert_allclose(getattr(verified,attribute),np.asarray(value[key],dtype=np.float64),
                                   rtol=1e-12,atol=0)
    assert int(verified.n_samples_seen_)==value['n_samples_seen']
    return verified


def restore_seed42(torch, folder, X, y, features, scaler):
    from pytorch_tabular import TabularModel
    from omegaconf import OmegaConf
    torch.manual_seed(42)
    config=OmegaConf.load(folder/'config.yml')
    assert list(config.continuous_cols)==features and config.task=='classification'
    assert config.seed==42 and config.load_best is False and config.checkpoints is None
    tab=TabularModel(config=config,verbose=False)
    tab.logger=False
    frame=pd.DataFrame(X,columns=features)
    frame['ml_label']=y.astype(np.int64)
    tab.datamodule=tab.prepare_dataloader(train=frame,validation=frame.iloc[:512].copy(),seed=42)
    tab.model=tab.prepare_model(tab.datamodule)
    tab.model.load_state_dict(torch.load(folder/'selected_state_dict.pt',map_location='cpu',weights_only=True))
    tab.model.cpu().eval()
    tab._is_fitted=True
    assert tab.datamodule.label_encoder[0].classes_.tolist()==list(range(24))
    assert len(tab.datamodule.train_dataset)==len(y)
    np.testing.assert_allclose(tab.datamodule.scaler.mean_,scaler.mean_,rtol=1e-12,atol=0)
    np.testing.assert_allclose(tab.datamodule.scaler.scale_,scaler.scale_,rtol=1e-12,atol=0)
    return tab


def verify_validation(torch, stage1, model, base, split, features, encoder, folder):
    parts=[]
    columns=list(dict.fromkeys(['source_record_id','ml_label',split]+features))
    for chunk in pd.read_csv(base/'o3_primary.csv.gz',usecols=columns,
                             float_precision='round_trip',chunksize=50000):
        parts.append(chunk.loc[chunk[split].eq('validation')].copy())
    val=pd.concat(parts,ignore_index=True)
    with np.load(folder/'validation_predictions.npz',allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['source_record_id'],val.source_record_id.astype(str).to_numpy())
        np.testing.assert_array_equal(saved['y_true'],encoder.transform(val.ml_label))
        prob,_=stage1.predict(torch,model,model.datamodule.scaler,val[features].to_numpy(dtype=np.float64))
        np.testing.assert_allclose(prob,saved['probabilities'],rtol=1e-5,atol=1e-6)
        np.testing.assert_array_equal(prob.argmax(axis=1),saved['y_pred'])
    print('  Seed42 restore verified against saved validation probabilities and predictions.',flush=True)


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
    source=Path(__file__).resolve().with_name('fttransformer_revision_stage1.py')
    assert sha(source)==EXPECTED_STAGE1_SHA==protocol['script_sha256'],'Restore unchanged FT-Transformer stage-1 script.'
    spec=importlib.util.spec_from_file_location('frozen_ft_stage1',source)
    stage1=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stage1)
    core=stage1.core
    versions=dict(python=platform.python_version(),torch=str(torch.__version__),
                  numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__,
                  pytorch_tabular=version('pytorch-tabular'),pytorch_lightning=version('pytorch-lightning'),
                  torchmetrics=version('torchmetrics'))
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
        assert selected['epochs']==20 and selected['activation']=='GEGLU'
        assert selected['weight_decay']==0.01 and selected['batch_size']==512
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
    frozen=dict(stage='FT-Transformer fixed three-seed test evaluation',seeds=SEEDS,
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
            config_path=folder/f'seed_{seed}_config.yml'
            if result_path.exists():
                result=read(result_path)
                assert result['frozen_test_protocol_sha256']==lock_hash
                assert result['seed']==seed and result['condition']==tag
                for name,expected in result['artifact_sha256'].items():
                    assert sha(folder/name)==expected,f'Completed artifact changed: {name}'
                with np.load(pred_path,allow_pickle=False) as saved:
                    assert np.array_equal(saved['true_label'],ye)
                    np.testing.assert_array_equal(saved['source_record_id'],metadata.source_record_id.astype(str).to_numpy())
                    pred=saved['prediction']
                for key,value in core.scores(ye,pred).items():
                    assert np.isclose(result['test_'+key],value,rtol=0,atol=1e-12)
                report=read(class_path)
                print(f'  seed {seed}: verified completed checkpoint',flush=True)
            else:
                if seed==42:
                    print('  seed42: reusing reviewed stage-1 model',flush=True)
                    model=restore_seed42(torch,previous/tag,Xt,yt,features,verified_scaler)
                    scaler=model.datamodule.scaler
                    verify_validation(torch,stage1,model,base,split,features,encoder,previous/tag)
                    fit_seconds=None
                    shutil.copy2(previous/tag/'selected_state_dict.pt',model_path)
                    shutil.copy2(previous/tag/'training_history.csv',history_path)
                else:
                    print(f'  seed{seed}: training {len(yt):,} rows for 20 epochs',flush=True)
                    model,scaler,history,fit_seconds=stage1.fit_model(torch,Xt,yt,selected['candidate'],seed,f'  seed{seed}',features)
                    np.testing.assert_allclose(scaler.mean_,verified_scaler.mean_,rtol=1e-12,atol=0)
                    np.testing.assert_allclose(scaler.scale_,verified_scaler.scale_,rtol=1e-12,atol=0)
                    model.save_weights(model_path)
                    pd.DataFrame(history).to_csv(history_path,index=False)
                    del history
                from omegaconf import OmegaConf
                OmegaConf.save(model.config,config_path,resolve=True)
                core.save_json(scaler_path,dict(features=features,mean=scaler.mean_.tolist(),
                    scale=scaler.scale_.tolist(),var=scaler.var_.tolist(),n_samples_seen=int(scaler.n_samples_seen_)))
                prob,pred_seconds=stage1.predict(torch,model,scaler,Xe)
                pred=prob.argmax(axis=1).astype(np.int16)
                np.savez_compressed(pred_path,prediction=pred,true_label=ye.astype(np.int16),
                                    source_record_id=metadata.source_record_id.astype(str).to_numpy(dtype=str))
                report=classification_report(ye,pred,labels=np.arange(24),target_names=encoder.classes_.tolist(),
                                             zero_division=0,output_dict=True)
                core.save_json(class_path,report)
                names=[model_path.name,scaler_path.name,history_path.name,pred_path.name,class_path.name,
                       'test_records.csv.gz','label_mapping.json',config_path.name]
                result=dict(condition=tag,split=split,feature_set=feature_set,seed=seed,
                            candidate=selected['candidate'],train_rows=len(yt),test_rows=len(ye),
                            test_supported_classes=int(len(np.unique(ye))),reused_stage1_model=seed==42,
                            fit_seconds=fit_seconds,test_prediction_batch_seconds=pred_seconds,
                            trainable_parameters=sum(p.numel() for p in model.model.parameters() if p.requires_grad),
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
    print('\nFT-TRANSFORMER STAGE 2 COMPLETE. Keep configurations fixed; do not retune from test scores.',flush=True)
    print(summary.to_string(index=False),flush=True)
    print('\nShare test_summary.csv, test_per_seed.csv, test_class_metrics.csv and timestamp_differences.csv from this FT-Transformer output folder.',flush=True)


if __name__=='__main__':
    main()
