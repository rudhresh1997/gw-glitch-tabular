"""Four-condition XGBoost search using outer training data only.

12 fixed candidates x 3 folds x 4 conditions = 144 CV fits, followed by
4 full-training fits and validation evaluation. No test evaluation.
Run in a fresh env1 Python process. Re-running resumes compatible checkpoints.

This is a new supplemental protocol, not a reproduction of historical 100-trial tuning.
Candidate selection uses the mean fold weighted F1 only. Macro F1 and accuracy are
diagnostics. Time folds may lack rare training labels: a fixed 24-output native
Booster handles them; counts and unseen validation labels are recorded, never hidden.
Random CV and forward CV estimate different generalization tasks; their CV means
are not paired measurements. No early stopping and no temporal embargo are used.
The optional frozen 32k training subsets supply all CV data. Chosen models are then
fit on the respective entire outer training partition and scored on outer validation.
No statistical significance or timestamp-causality claim follows from this stage.

Model settings use raw float32 inputs and normalized inverse-frequency weights,
recomputed within each training fold. Absolute peak_time loses precision under
float32 as in the legacy unscaled XGBoost representation; the diagnostic file
quantifies training unique-value loss. Do not silently interpret it as a benchmark
of optimally represented time. The six-feature arm omits all timing columns.

Fit timing includes DMatrix construction but no evaluation loop; it differs from
the pilot timing definition. Process peak memory is cumulative across this run,
not per-model memory. Batch prediction time excludes DMatrix construction and is
not single-event latency. This stage's timing is for scheduling, not publication.
Test sets stay unscored. Final seeds and test evaluation require a separate frozen
evaluation step. Do not revise this grid in response to test results.
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

for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[name] = '2'

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedGroupKFold, TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

CONDITIONS = [('random_split','original_nine'),('random_split','timestamp_free_six'),
              ('chronological_split','original_nine'),('chronological_split','timestamp_free_six')]
# rounds, depth, learning rate, column fraction, row fraction; frozen before scoring.
GRID = [(100,6,.10,1.,1.),(200,6,.05,1.,1.),(250,6,.10,.8,.8),
        (150,4,.10,1.,1.),(250,4,.05,.8,.8),(200,4,.10,.8,1.),
        (100,8,.10,.8,.8),(200,8,.05,.8,.8),(150,8,.10,1.,.8),
        (250,5,.05,1.,.8),(150,5,.10,.8,1.),(250,6,.05,.8,1.)]


def save_json(path, value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    os.replace(tmp,path)


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def peak_mib():
    v=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(v/(1024**2 if sys.platform=='darwin' else 1024))


def make_folds(frame, y, protocol):
    if protocol=='random_split':
        splitter=StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=42)
        folds=list(splitter.split(np.zeros(len(y)),y,groups=frame.group_id))
    else:
        # Unique timestamp blocks keep simultaneous events and repeated groups together.
        assert frame.groupby('group_id').peak_gps_ns.nunique().max()==1
        times=np.sort(frame.peak_gps_ns.unique())
        folds=[]
        for a,b in TimeSeriesSplit(n_splits=3).split(times):
            i=np.flatnonzero(frame.peak_gps_ns.isin(times[a]).to_numpy())
            j=np.flatnonzero(frame.peak_gps_ns.isin(times[b]).to_numpy())
            assert frame.iloc[i].peak_gps_ns.max()<frame.iloc[j].peak_gps_ns.min()
            folds.append((i,j))
    for i,j in folds:
        assert len(i)>0 and len(j)>0
        assert set(frame.iloc[i].group_id).isdisjoint(frame.iloc[j].group_id)
    return folds


def parameters(candidate):
    rounds,depth,lr,col,row=GRID[candidate]
    return rounds,dict(objective='multi:softprob',num_class=24,tree_method='hist',
                       device='cpu',nthread=2,seed=42,eval_metric='mlogloss',
                       max_depth=depth,eta=lr,colsample_bytree=col,subsample=row,
                       max_bin=256,min_child_weight=1,reg_lambda=1.,reg_alpha=0.,
                       verbosity=1)


def fit_model(xgb, X, y, candidate, feature_names):
    rounds,params=parameters(candidate)
    w=compute_sample_weight('balanced',y).astype(np.float32)
    t=time.perf_counter()
    dm=xgb.DMatrix(X,label=y,weight=w,feature_names=feature_names,nthread=2)
    model=xgb.train(params,dm,num_boost_round=rounds,verbose_eval=False)
    return model,time.perf_counter()-t


def scores(y, pred):
    return dict(weighted_f1=float(f1_score(y,pred,average='weighted',zero_division=0)),
                macro_f1_all24=float(f1_score(y,pred,labels=np.arange(24),average='macro',zero_division=0)),
                macro_f1_supported=float(f1_score(y,pred,labels=np.unique(y),average='macro',zero_division=0)),
                accuracy=float(accuracy_score(y,pred)))


def main():
    import xgboost as xgb
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-dir',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,required=True)
    args=ap.parse_args()
    base=args.data_dir.resolve();out=args.output_dir.resolve()
    out.mkdir(parents=True,exist_ok=True)
    needed=['manifest.json','o3_primary.csv.gz','split_summary.csv',
            'tuning_random_split.csv.gz','tuning_chronological_split.csv.gz']
    expected=dict(line.split(maxsplit=1) for line in (base/'SHA256SUMS.txt').read_text().splitlines())
    expected={v.strip():k for k,v in expected.items()}
    hashes={}
    print('Checking package integrity...',flush=True)
    for name in needed:
        hashes[name]=sha(base/name)
        assert hashes[name]==expected[name],f'Checksum mismatch: {name}'
    spec=json.loads((base/'manifest.json').read_text())
    protocol=dict(stage='training-only CV plus outer validation; test not evaluated',
                  version='1.0',script_sha256=sha(Path(__file__)),inputs=hashes,
                  candidate_grid=GRID,grid_columns=['rounds','max_depth','eta','colsample_bytree','subsample'],
                  conditions=CONDITIONS,folds=3,selection='highest mean fold weighted F1; ties choose earlier candidate',
                  cv_random='StratifiedGroupKFold, seed42',cv_temporal='3 expanding timestamp-block folds; no embargo',
                  class_weight_rule='fold-training N/(number of present classes * class count)',
                  numeric_representation='float32, unscaled raw columns; no timestamp centering',
                  features=spec['feature_sets'],seed=42,threads=2,
                  versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,
                                sklearn=sklearn.__version__,xgboost=xgb.__version__))
    # Normalize tuples to the JSON representation for restart equality.
    protocol=json.loads(json.dumps(protocol))
    cfg=out/'protocol.json'
    if cfg.exists():
        assert json.loads(cfg.read_text())==protocol,'Existing run has a different protocol. Use a new output directory.'
    else:
        save_json(cfg,protocol)
    summary_rows=[]
    split_summary=pd.read_csv(base/'split_summary.csv')
    for split,feature_set in CONDITIONS:
        tag=f'{split}__{feature_set}'
        folder=out/tag;folder.mkdir(exist_ok=True)
        print('\nCONDITION: '+tag,flush=True)
        final=folder/'validation_result.json'
        if final.exists():
            result=json.loads(final.read_text())
            assert (folder/'selected_model.ubj').is_file()
            summary_rows.append(result)
            print('Already completed; using saved result.',flush=True)
            continue
        features=spec['feature_sets'][feature_set]
        members=pd.read_csv(base/f'tuning_{split}.csv.gz')
        ids=set(members.source_record_id);group_ids=set(members.group_id)
        cols=['source_record_id','group_id','peak_gps_ns','ml_label',split]+features
        cols=list(dict.fromkeys(cols))
        train_parts=[];val_parts=[]
        for chunk in pd.read_csv(base/'o3_primary.csv.gz',usecols=cols,chunksize=50000,float_precision='round_trip'):
            tr=chunk[split].eq('train');va=chunk[split].eq('validation')
            assert chunk.loc[chunk.source_record_id.isin(ids),split].eq('train').all()
            if tr.any():train_parts.append(chunk.loc[tr].copy())
            if va.any():val_parts.append(chunk.loc[va].copy())
        train=pd.concat(train_parts,ignore_index=True);validation=pd.concat(val_parts,ignore_index=True)
        del train_parts,val_parts,chunk
        for name,frame in [('train',train),('validation',validation)]:
            n=int(split_summary.loc[(split_summary.configuration==split)&(split_summary.partition==name),'rows'].iloc[0])
            assert len(frame)==n and frame.source_record_id.is_unique
        assert set(train.group_id).isdisjoint(validation.group_id)
        selected=train.source_record_id.isin(ids)
        assert selected.equals(train.group_id.isin(group_ids))
        tune=train.loc[selected].reset_index(drop=True)
        assert set(tune.source_record_id)==ids
        encoder=LabelEncoder().fit(train.ml_label)
        assert len(encoder.classes_)==24
        y=encoder.transform(tune.ml_label)
        X=tune[features].to_numpy(dtype=np.float32)
        assert np.isfinite(X).all()
        folds=make_folds(tune,y,split)
        fold_diagnostics=[]
        for k,(i,j) in enumerate(folds):
            missing=sorted(set(range(24))-set(y[i]))
            fold_diagnostics.append(dict(fold=k,train_rows=len(i),validation_rows=len(j),
                training_missing_classes=encoder.inverse_transform(missing).tolist() if missing else [],
                validation_classes=int(len(np.unique(y[j]))),
                validation_rows_with_unseen_training_label=int((~np.isin(y[j],y[i])).sum())))
        save_json(folder/'fold_diagnostics.json',fold_diagnostics)
        # Persist exact CV membership so it is not dependent on future sklearn versions.
        fold_path=folder/'cv_membership.csv.gz'
        if not fold_path.exists():
            records=[]
            for k,(i,j) in enumerate(folds):
                for role,ix in [('train',i),('validation',j)]:
                    records.append(tune.iloc[ix][['source_record_id','group_id']].assign(fold=k,role=role))
            pd.concat(records,ignore_index=True).to_csv(fold_path,index=False,compression='gzip')
        trials=[]
        for candidate in range(len(GRID)):
            fold_results=[]
            for k,(i,j) in enumerate(folds):
                path=folder/f'candidate_{candidate:02d}_fold_{k}.json'
                if path.exists():
                    result=json.loads(path.read_text())
                else:
                    model,seconds=fit_model(xgb,X[i],y[i],candidate,features)
                    dv=xgb.DMatrix(X[j],feature_names=features,nthread=2)
                    pred=np.argmax(model.predict(dv),axis=1)
                    result=dict(candidate=candidate,fold=k,fit_seconds=seconds,**scores(y[j],pred))
                    save_json(path,result)
                    del model,dv,pred;gc.collect()
                fold_results.append(result)
                print(f'  candidate {candidate+1:02d}/12 fold {k+1}/3: F1={result["weighted_f1"]:.5f}, fit={result["fit_seconds"]:.1f}s',flush=True)
            trials.append(dict(candidate=candidate,cv_mean_weighted_f1=float(np.mean([r['weighted_f1'] for r in fold_results])),
                               cv_fit_seconds=sum(r['fit_seconds'] for r in fold_results)))
            pd.DataFrame(trials).to_csv(folder/'cv_results.csv',index=False)
        best=max(trials,key=lambda r:r['cv_mean_weighted_f1'])
        choice=best['candidate']
        rounds,params=parameters(choice)
        save_json(folder/'selected_configuration.json',dict(**best,rounds=rounds,params=params,
                  features=features,labels=encoder.classes_.tolist(),selection_used_outer_validation=False))
        print(f'Selected candidate {choice+1}; fitting all {len(train):,} training rows.',flush=True)
        if 'peak_time' in features:
            save_json(folder/'timestamp_precision.json',dict(
                train_distinct_peak_seconds=int(train.peak_time.nunique()),
                distinct_after_float32=int(train.peak_time.astype(np.float32).nunique()),
                note='Raw absolute seconds lose resolution under float32, matching the archived unscaled representation. Timestamp-free condition removes all three timing columns.'))
        del X,tune,folds;gc.collect()
        Xt=train[features].to_numpy(dtype=np.float32);yt=encoder.transform(train.ml_label)
        Xv=validation[features].to_numpy(dtype=np.float32);yv=encoder.transform(validation.ml_label)
        assert set(validation.ml_label)<=set(encoder.classes_)
        assert np.isfinite(Xt).all() and np.isfinite(Xv).all()
        validation_ids=validation.source_record_id.to_numpy()
        del train,validation;gc.collect()
        model,seconds=fit_model(xgb,Xt,yt,choice,features)
        dv=xgb.DMatrix(Xv,feature_names=features,nthread=2)
        t=time.perf_counter();prob=model.predict(dv);prediction_seconds=time.perf_counter()-t
        pred=np.argmax(prob,axis=1)
        model.save_model(folder/'selected_model.ubj')
        pd.DataFrame(dict(source_record_id=validation_ids,true_label=encoder.inverse_transform(yv),
                          predicted_label=encoder.inverse_transform(pred))).to_csv(folder/'validation_predictions.csv.gz',index=False,compression='gzip')
        result=dict(condition=tag,split=split,feature_set=feature_set,candidate=choice,
                    cv_mean_weighted_f1=best['cv_mean_weighted_f1'],train_rows=len(yt),validation_rows=len(yv),
                    validation_supported_classes=int(len(np.unique(yv))),fit_seconds=seconds,
                    validation_prediction_batch_seconds=prediction_seconds,
                    process_lifetime_peak_rss_mib=peak_mib(),
                    model_mib=(folder/'selected_model.ubj').stat().st_size/1024**2,
                    test_evaluated=False,**{'validation_'+k:v for k,v in scores(yv,pred).items()})
        save_json(final,result)
        summary_rows.append(result)
        pd.DataFrame(summary_rows).to_csv(out/'validation_summary.csv',index=False)
        del model,dv,Xt,Xv,yt,yv,prob,pred;gc.collect()
        print(json.dumps(result,indent=2),flush=True)
    pd.DataFrame(summary_rows).to_csv(out/'validation_summary.csv',index=False)
    print('\nSTAGE 1 COMPLETE — no test evaluation.',flush=True)
    print(pd.DataFrame(summary_rows)[['condition','cv_mean_weighted_f1','validation_weighted_f1','validation_macro_f1_all24','fit_seconds']].to_string(index=False),flush=True)
    print('\nShare validation_summary.csv and protocol.json. Keep all checkpoint folders.',flush=True)


if __name__=='__main__':
    main()
