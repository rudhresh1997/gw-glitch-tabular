"""Frozen three-seed test evaluation for the completed JPCO stage-1 run.

Requires xgboost_revision_stage1.py beside this script, unchanged from stage 1.
Reuse seed 42; train seeds 43 and 44 with frozen configurations and outer TRAIN only.
No tuning, early stopping, model selection or train+validation refit occurs here.
Reported sample SD (ddof=1) describes variation across 3 training seeds on a fixed
test partition, not a confidence interval or uncertainty from drawing new events.
Test sets differ across split protocols; cross-protocol metrics are not paired.
Paired timestamp deltas within each protocol use identical test records/seeds.
Raw timestamp float32 representation is retained, with the stage-1 limitation.
"""
import argparse
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import sys
import platform
import time
import gc

for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[name]='2'

import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_sample_weight

SEEDS=[42,43,44]
EXPECTED_PROTOCOL_SHA='675d91f3dc6f2b25148e8f4e3e52d5a88a021e1619377f007f1d3a9942fe624d'
EXPECTED_CHOICES={
    'random_split__original_nine':7,
    'random_split__timestamp_free_six':8,
    'chronological_split__original_nine':7,
    'chronological_split__timestamp_free_six':2,
}


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def save(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    os.replace(tmp,path)


def validate_cv(folder,choice):
    """Recompute the training-only selection from all 36 saved fold checkpoints."""
    means=[]
    for c in range(12):
        scores=[]
        for fold in range(3):
            r=json.loads((folder/f'candidate_{c:02d}_fold_{fold}.json').read_text())
            assert r['candidate']==c and r['fold']==fold
            v=float(r['weighted_f1'])
            assert np.isfinite(v) and 0<=v<=1
            scores.append(v)
        means.append(float(np.mean(scores)))
    assert int(np.argmax(means))==choice,'Saved CV checkpoints disagree with frozen choice.'
    return means[choice]


def main():
    import xgboost as xgb
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-dir',type=Path,required=True)
    ap.add_argument('--stage1-dir',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,required=True)
    args=ap.parse_args()
    base=args.data_dir.resolve();previous=args.stage1_dir.resolve();out=args.output_dir.resolve()
    protocol_path=previous/'protocol.json'
    assert digest(protocol_path)==EXPECTED_PROTOCOL_SHA,'Stage-1 protocol differs from the reviewed run.'
    protocol=json.loads(protocol_path.read_text())
    source=Path(__file__).resolve().with_name('xgboost_revision_stage1.py')
    assert digest(source)==protocol['script_sha256'],'Restore the unchanged stage-1 script beside this file.'
    spec=importlib.util.spec_from_file_location('jpco_stage1',source)
    stage1=importlib.util.module_from_spec(spec);spec.loader.exec_module(stage1)
    versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,
                  sklearn=sklearn.__version__,xgboost=xgb.__version__)
    assert versions==protocol['versions'],'Use the same env1 versions as stage 1.'
    print('Checking frozen inputs, configurations and CV selections...',flush=True)
    for name,expected in protocol['inputs'].items():
        assert digest(base/name)==expected,f'Changed input file: {name}'
    dataset=json.loads((base/'manifest.json').read_text())
    choices={};preflight={}
    for split,featureset in stage1.CONDITIONS:
        tag=f'{split}__{featureset}';folder=previous/tag
        selected=json.loads((folder/'selected_configuration.json').read_text())
        choice=EXPECTED_CHOICES[tag]
        assert selected['candidate']==choice and not selected['selection_used_outer_validation']
        assert selected['features']==dataset['feature_sets'][featureset]
        rounds,params=stage1.parameters(choice)
        assert selected['rounds']==rounds and selected['params']==params
        cv=validate_cv(folder,choice)
        assert np.isclose(cv,selected['cv_mean_weighted_f1'],rtol=0,atol=1e-12)
        assert not json.loads((folder/'validation_result.json').read_text())['test_evaluated']
        choices[tag]=selected
        preflight[tag]=dict(candidate=choice,rounds=rounds,params=params,features=selected['features'],
                            seed42_model_sha256=digest(folder/'selected_model.ubj'),
                            selected_configuration_sha256=digest(folder/'selected_configuration.json'))
    frozen=dict(stage='test evaluation; selection frozen before test scoring',seeds=SEEDS,
                stage1_protocol_sha256=EXPECTED_PROTOCOL_SHA,script_sha256=digest(Path(__file__)),
                inputs=protocol['inputs'],selections=preflight,versions=versions,
                training_scope='outer training only; no validation refit',
                metrics=['weighted_f1','macro_f1_all24','macro_f1_supported','accuracy'],
                aggregation='arithmetic mean and sample SD (ddof=1) across 3 seeds; not a confidence interval')
    out.mkdir(parents=True,exist_ok=True)
    lock=out/'frozen_test_protocol.json'
    if lock.exists():
        assert json.loads(lock.read_text())==frozen,'Incompatible existing test run. Do not change the frozen protocol.'
    else:
        save(lock,frozen)
    all_results=[];all_class_rows=[]
    split_summary=pd.read_csv(base/'split_summary.csv')
    for split,featureset in stage1.CONDITIONS:
        tag=f'{split}__{featureset}';folder=out/tag;folder.mkdir(exist_ok=True)
        selected=choices[tag];features=selected['features']
        print('\nTEST CONDITION: '+tag,flush=True)
        # Load only training/test subsets for this experiment. No test-based transformations.
        cols=list(dict.fromkeys(['source_record_id','group_id','peak_gps_ns','ml_label',split]+features))
        a=[];b=[]
        for chunk in pd.read_csv(base/'o3_primary.csv.gz',usecols=cols,float_precision='round_trip',chunksize=50000):
            tr=chunk[split].eq('train');te=chunk[split].eq('test')
            if tr.any():a.append(chunk.loc[tr].copy())
            if te.any():b.append(chunk.loc[te].copy())
        train=pd.concat(a,ignore_index=True);test=pd.concat(b,ignore_index=True)
        del a,b,chunk
        for partition,frame in [('train',train),('test',test)]:
            n=int(split_summary.loc[(split_summary.configuration==split)&(split_summary.partition==partition),'rows'].iloc[0])
            assert len(frame)==n and frame.source_record_id.is_unique
        assert set(train.group_id).isdisjoint(test.group_id)
        if split=='chronological_split':assert train.peak_gps_ns.max()<test.peak_gps_ns.min()
        encoder=LabelEncoder().fit(train.ml_label)
        assert encoder.classes_.tolist()==selected['labels'] and len(encoder.classes_)==24
        assert set(test.ml_label)<=set(train.ml_label)
        Xt=train[features].to_numpy(dtype=np.float32);yt=encoder.transform(train.ml_label)
        Xe=test[features].to_numpy(dtype=np.float32);ye=encoder.transform(test.ml_label)
        assert np.isfinite(Xt).all() and np.isfinite(Xe).all()
        metadata=test[['source_record_id','group_id','peak_gps_ns','ml_label']].reset_index(drop=True)
        metadata.to_csv(folder/'test_records.csv.gz',index=False,compression='gzip')
        save(folder/'label_mapping.json',encoder.classes_.tolist())
        del train,test;gc.collect()
        dt=xgb.DMatrix(Xt,label=yt,weight=compute_sample_weight('balanced',yt).astype(np.float32),
                       feature_names=features,nthread=2)
        de=xgb.DMatrix(Xe,feature_names=features,nthread=2)
        for seed in SEEDS:
            result_path=folder/f'seed_{seed}_test_metrics.json'
            pred_path=folder/f'seed_{seed}_predictions.npz'
            class_path=folder/f'seed_{seed}_class_report.json'
            if result_path.exists():
                result=json.loads(result_path.read_text())
                assert result['seed']==seed and result['condition']==tag
                pred=np.load(pred_path,allow_pickle=False)['prediction']
                assert len(pred)==len(ye)
                report=json.loads(class_path.read_text())
                print(f'  seed {seed}: completed checkpoint',flush=True)
            else:
                params=dict(selected['params']);params['seed']=seed
                if seed==42:
                    model=xgb.Booster(params={'nthread':2,'device':'cpu'})
                    model.load_model(previous/tag/'selected_model.ubj')
                    fit_seconds=None
                    model_path=previous/tag/'selected_model.ubj'
                    print('  seed 42: reusing frozen stage-1 model',flush=True)
                else:
                    print(f'  seed {seed}: training {len(yt):,} rows',flush=True)
                    t=time.perf_counter()
                    model=xgb.train(params,dt,num_boost_round=selected['rounds'],verbose_eval=False)
                    fit_seconds=time.perf_counter()-t
                    model_path=folder/f'seed_{seed}_model.ubj';model.save_model(model_path)
                assert model.feature_names==features
                assert model.num_boosted_rounds()==selected['rounds']
                t=time.perf_counter();prob=model.predict(de);predict_seconds=time.perf_counter()-t
                assert prob.shape==(len(ye),24) and np.isfinite(prob).all()
                pred=np.argmax(prob,axis=1).astype(np.int16)
                # Align predictions with test_records.csv.gz; labels remain in label_mapping.json.
                np.savez_compressed(pred_path,prediction=pred,true_label=ye.astype(np.int16))
                report=classification_report(ye,pred,labels=np.arange(24),target_names=encoder.classes_.tolist(),
                                             output_dict=True,zero_division=0)
                save(class_path,report)
                result=dict(condition=tag,split=split,feature_set=featureset,seed=seed,
                            candidate=selected['candidate'],train_rows=len(yt),test_rows=len(ye),
                            test_supported_classes=int(len(np.unique(ye))),reused_stage1_model=seed==42,
                            fit_seconds=fit_seconds,test_prediction_batch_seconds=predict_seconds,
                            model_sha256=digest(model_path),model_mib=model_path.stat().st_size/1024**2,
                            process_lifetime_peak_rss_mib=stage1.peak_mib(),test_evaluated=True,
                            **{'test_'+k:v for k,v in stage1.scores(ye,pred).items()})
                save(result_path,result)
                del model,prob;gc.collect()
                print(f'  seed {seed}: test weighted F1={result["test_weighted_f1"]:.6f}',flush=True)
            all_results.append(result)
            for label in encoder.classes_:
                r=report[label]
                all_class_rows.append(dict(condition=tag,seed=seed,label=label,support=int(r['support']),
                                            precision=r['precision'],recall=r['recall'],f1=r['f1-score']))
            pd.DataFrame(all_results).to_csv(out/'test_per_seed.csv',index=False)
        del Xt,Xe,yt,ye,dt,de,metadata,pred;gc.collect()
    raw=pd.DataFrame(all_results)
    rows=[]
    for tag,g in raw.groupby('condition',sort=False):
        assert set(g.seed)==set(SEEDS)
        r=dict(condition=tag,train_rows=int(g.train_rows.iloc[0]),test_rows=int(g.test_rows.iloc[0]),
               supported_classes=int(g.test_supported_classes.iloc[0]),seeds=len(g))
        for metric in frozen['metrics']:
            r[metric+'_mean']=float(g['test_'+metric].mean())
            r[metric+'_sd']=float(g['test_'+metric].std(ddof=1))
        rows.append(r)
    summary=pd.DataFrame(rows);summary.to_csv(out/'test_summary.csv',index=False)
    pd.DataFrame(all_class_rows).to_csv(out/'test_class_metrics.csv',index=False)
    differences=[]
    for split in ['random_split','chronological_split']:
        a=raw[(raw.split==split)&(raw.feature_set=='original_nine')].set_index('seed')
        b=raw[(raw.split==split)&(raw.feature_set=='timestamp_free_six')].set_index('seed')
        left=pd.read_csv(out/f'{split}__original_nine'/'test_records.csv.gz',usecols=['source_record_id'])
        right=pd.read_csv(out/f'{split}__timestamp_free_six'/'test_records.csv.gz',usecols=['source_record_id'])
        assert left.equals(right)
        diff=(a.test_weighted_f1-b.test_weighted_f1)*100
        differences.append(dict(split=split,nine_minus_six_weighted_f1_pp_mean=float(diff.mean()),
                                 nine_minus_six_weighted_f1_pp_sd=float(diff.std(ddof=1))))
    pd.DataFrame(differences).to_csv(out/'timestamp_differences.csv',index=False)
    print('\nSTAGE 2 COMPLETE: fixed test results; do not retune against these scores.',flush=True)
    print(summary.to_string(index=False),flush=True)
    print('\nShare test_summary.csv, test_per_seed.csv, timestamp_differences.csv, and test_class_metrics.csv.',flush=True)


if __name__=='__main__':
    main()
