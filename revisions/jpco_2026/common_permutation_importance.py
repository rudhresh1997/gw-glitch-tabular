"""JPCO common permutation importance for 36 frozen revision models.

No neural/tree training, tuning, feature selection or modification of prior outputs.
Reconstruct training-only preprocessing and verify it against saved values.
Restore all test predictions before explaining each model. Explain one fixed,
label-blind sample of 5000 whole groups per test split, including all records in
selected groups. Sampling, record order and permutations are shared across
models, feature sets and training seeds. Different splits have different cohorts.

Ten marginal single-column permutations per feature. Primary importance is the
signed drop in weighted F1; secondary is the signed drop in macro F1 over all
24 labels. Do not clip negatives, take absolute values or normalize these values.
Shuffling correlated features can produce implausible combinations and mask
shared reliance. This is descriptive score-based importance, not causal physics.
The cohort is one fixed sample; repeat SD is shuffle variability, not uncertainty
from drawing another cohort. Training-seed SD is reported separately (n=3).

Spearman rank correlations use average ranks for ties. Constant-vector results
are undefined. Six/nine features and three training seeds do not justify an
asymptotic significance claim. No p values or confidence intervals are generated.
This common-method analysis alone does not prove TreeSHAP/IG agreement.

Reference: https://scikit-learn.org/stable/modules/permutation_importance.html
Run as a fresh env1 subprocess beside the six unchanged stage1/stage2 scripts.
Completed model-feature results resume under an identical protocol.
"""
import argparse
import gc
import hashlib
import importlib.util
from importlib.metadata import version
import itertools
import json
import os
from pathlib import Path
import platform
import time

for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[name]='2'

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

MODELS=['XGBoost','MLP','FT-Transformer']
SEEDS=[42,43,44]
SPLITS=['random_split','chronological_split']
FEATURE_SETS=['original_nine','timestamp_free_six']
METRICS=['weighted_f1','macro_f1_all24']
GROUPS=5000
REPEATS=10
SOURCE_HASHES = {'mlp_revision_stage1.py': '1aed088a3fa04eaed98c5fa081328b606a3085b43c2b3bd71a44bf1f14d18b93', 'xgboost_revision_stage1.py': '7c06aa5cffb280dea07ec88ce4c140e9d6a98db6574996b60a377d0946c78b7b', 'fttransformer_revision_stage2.py': 'aa6959a300e51b6284a5278b457a166138ed0bee68ca32bb583810c15d8f6d05', 'fttransformer_revision_stage1.py': 'bf5dd746c97db114a3054416adb46e4d7df79a056d6ea74adfaf4931bcf07af3', 'mlp_revision_stage2.py': '4e9be180bf7db4efc8351d9430572dc935086aaf6af4ebfaf80a16ac77522d70', 'xgboost_revision_stage2.py': '37894c10133e1d0e00edea4b5957254a1a871933ab9a992df0e9ae0d7e233e7f'}
REVIEWED = {'XGBoost': {'test_summary.csv': 'b7f19736473a68336a1716d3f178592d3809ec7dd049b71529ad71036e1e122a', 'test_per_seed.csv': 'f330921e6209deb24c31af2a3053e19df18cef53c5d4cb8f2333871feeb6ce72', 'test_class_metrics.csv': '5d5e652d8cbc5fb7034821c5bdc082152408745e842f7016f4ce43fcc86f13a5', 'timestamp_differences.csv': '223f3a0a48d0e3607f7076a84e2be4b18305a80afec9d171c20c72e8ca79bc96'}, 'MLP': {'test_summary.csv': '12cef3b0cfcecd82e1c97431c1f50a9d766a3e168dc4765a87a847a893a291ef', 'test_per_seed.csv': 'ccc026e3d11c8bb9a699b698f434c853b345bc20d7f20b3be7233c8d3c5c5b15', 'test_class_metrics.csv': '4645f9455c753a326657036c282c8a7526a28e8e496c378e74f9cf25da302717', 'timestamp_differences.csv': 'ad6f601c00c0e688f831597f885de91d20085b36d40f92664e1834854a80df29'}, 'FT-Transformer': {'test_summary.csv': '6b6c82fc87e8a13b1e83434cb780c148b6255d794e01c6a01467641f00601517', 'test_per_seed.csv': '84ef06cf62d6d7f9c849d834b73e2f53d40ef9db80763f28361d41cc1c868dc0', 'test_class_metrics.csv': '8f25c24066207be7a7d4a069b6ba609cd536cf51e1c26d4613c59d34bc965e23', 'timestamp_differences.csv': '8ce99f9941ddb80989290340bbcf7806cedd361596e2790af261ce05365f86a7'}}
STAGE1_PROTOCOLS = {'XGBoost': '675d91f3dc6f2b25148e8f4e3e52d5a88a021e1619377f007f1d3a9942fe624d', 'MLP': '2cd30e32ca65a3f6f33633da2ec873516ff804b1faf41f9b996a3f206129a801', 'FT-Transformer': 'c2df0ad2ad895d722b803392616823ce22cc6f60a199e777346a3fd53ccc0dd4'}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path,value):
    path=Path(path)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    os.replace(temp,path)


def stable_key(*parts):
    return hashlib.sha256('|'.join(map(str,parts)).encode()).hexdigest()


def cohort_indices(test,split):
    groups=sorted(test.group_id.astype(str).unique(),key=lambda g:stable_key('jpco-pi-v1-cohort',split,g))
    assert len(groups)>=GROUPS
    chosen=set(groups[:GROUPS])
    return np.flatnonzero(test.group_id.astype(str).isin(chosen).to_numpy())


def permutation_indices(n,split,feature,repeat):
    seed=int(stable_key('jpco-pi-v1-shuffle',split,feature,repeat)[:16],16)
    return np.random.Generator(np.random.PCG64(seed)).permutation(n)


def scores(y,pred):
    return dict(weighted_f1=float(f1_score(y,pred,average='weighted',zero_division=0)),
                macro_f1_all24=float(f1_score(y,pred,labels=np.arange(24),average='macro',zero_division=0)),
                accuracy=float(accuracy_score(y,pred)))


def rank_correlation(a,b):
    a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
    assert a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    ra=pd.Series(a).rank(method='average').to_numpy()
    rb=pd.Series(b).rank(method='average').to_numpy()
    if np.ptp(ra)==0 or np.ptp(rb)==0:
        return None
    return float(np.corrcoef(ra,rb)[0,1])


def unique_path(paths,description):
    paths=sorted(set(p.resolve() for p in paths))
    if len(paths)!=1:
        raise RuntimeError(f'Expected one {description}, found {len(paths)}: {paths}. Share this message.')
    return paths[0]


def discover(root):
    stage1_files=list(root.rglob('protocol.json'))
    stage2_files=list(root.rglob('frozen_test_protocol.json'))
    found={}
    for model in MODELS:
        h=STAGE1_PROTOCOLS[model]
        first=unique_path([p.parent for p in stage1_files if sha(p)==h],model+' stage1 folder')
        second=[]
        for p in stage2_files:
            value=read(p)
            summary=p.parent/'test_summary.csv'
            if value.get('stage1_protocol_sha256')==h and summary.is_file() and sha(summary)==REVIEWED[model]['test_summary.csv']:
                second.append(p.parent)
        last=unique_path(second,model+' reviewed stage2 folder')
        found[model]={'stage1':first,'stage2':last}
        print(f'{model}: {last}',flush=True)
    return found


def import_helpers(root):
    modules={}
    for name,expected in SOURCE_HASHES.items():
        path=root/name
        assert path.is_file() and sha(path)==expected,f'Missing or modified helper: {path}'
    for key,name in [('mlp1','mlp_revision_stage1.py'),('mlp2','mlp_revision_stage2.py'),
                     ('ft1','fttransformer_revision_stage1.py')]:
        spec=importlib.util.spec_from_file_location('pi_'+key,root/name)
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules[key]=module
    return modules


def preflight(found,current_versions):
    artifacts={}
    cache={}
    def track(path):
        path=Path(path).resolve()
        if str(path) not in cache:
            cache[str(path)]=sha(path)
        artifacts[str(path)]=cache[str(path)]
        return cache[str(path)]
    entries={}
    input_hashes=None
    for model in MODELS:
        first,last=found[model]['stage1'],found[model]['stage2']
        stage1=read(first/'protocol.json')
        frozen=read(last/'frozen_test_protocol.json')
        track(first/'protocol.json'); frozen_hash=track(last/'frozen_test_protocol.json')
        for key,expected in frozen['versions'].items():
            assert current_versions[key]==expected,f'{model} package changed: {key}'
        if input_hashes is None:
            input_hashes=frozen['inputs']
        assert input_hashes==frozen['inputs']
        for filename,expected in REVIEWED[model].items():
            assert track(last/filename)==expected,f'Reviewed result changed: {last/filename}'
        seed_table=pd.read_csv(last/'test_per_seed.csv')
        for split in SPLITS:
            for featureset in FEATURE_SETS:
                tag=f'{split}__{featureset}'
                selected=read(first/tag/'selected_configuration.json')
                assert track(first/tag/'selected_configuration.json')==frozen['selections'][tag]['selected_configuration_sha256']
                assert selected['candidate']==frozen['selections'][tag]['candidate']
                assert selected['features']==stage1['features'][featureset]
                folder=last/tag
                metadata=folder/'test_records.csv.gz'; labels=folder/'label_mapping.json'
                track(metadata); track(labels)
                for seed in SEEDS:
                    result_path=folder/f'seed_{seed}_test_metrics.json'
                    r=read(result_path);track(result_path)
                    assert r['condition']==tag and r['seed']==seed and r['test_evaluated']
                    assert r['candidate']==selected['candidate']
                    if 'frozen_test_protocol_sha256' in r:
                        assert r['frozen_test_protocol_sha256']==frozen_hash
                    for filename,h in r.get('artifact_sha256',{}).items():
                        assert track(folder/filename)==h,f'Artifact changed: {folder/filename}'
                    row=seed_table[(seed_table.condition==tag)&(seed_table.seed==seed)]
                    assert len(row)==1
                    assert row.iloc[0].model_sha256==r['model_sha256']
                    assert row.iloc[0].candidate==r['candidate']
                    if 'frozen_test_protocol_sha256' in row.columns:
                        assert row.iloc[0].frozen_test_protocol_sha256==frozen_hash
                    for metric in METRICS+['accuracy']:
                        assert np.isclose(row.iloc[0]['test_'+metric],r['test_'+metric],rtol=0,atol=1e-12)
                    pred_path=folder/f'seed_{seed}_predictions.npz';track(pred_path)
                    if model=='XGBoost':
                        model_path=(first/tag/'selected_model.ubj') if seed==42 else folder/f'seed_{seed}_model.ubj'
                        extra={}
                    else:
                        model_path=folder/f'seed_{seed}_state_dict.pt'
                        extra={'scaler':str(folder/f'seed_{seed}_scaler.json')}
                        track(extra['scaler'])
                        if model=='FT-Transformer':
                            extra['config']=str(folder/f'seed_{seed}_config.yml');track(extra['config'])
                    assert track(model_path)==r['model_sha256']
                    key=f'{model}__{tag}__seed_{seed}'
                    entries[key]=dict(model=model,condition=tag,split=split,feature_set=featureset,
                        seed=seed,model_path=str(model_path),model_sha256=r['model_sha256'],
                        predictions=str(pred_path),metadata=str(metadata),labels=str(labels),
                        selected=selected,test_metrics=r,**extra)
    return entries,artifacts,input_hashes


def load_partitions(base,split,features):
    columns=list(dict.fromkeys(['source_record_id','group_id','peak_gps_ns','ml_label',split]+features))
    train_parts,test_parts=[],[]
    for chunk in pd.read_csv(base/'o3_primary.csv.gz',usecols=columns,float_precision='round_trip',chunksize=50000):
        train_parts.append(chunk.loc[chunk[split].eq('train')].copy())
        test_parts.append(chunk.loc[chunk[split].eq('test')].copy())
    train,test=pd.concat(train_parts,ignore_index=True),pd.concat(test_parts,ignore_index=True)
    counts=pd.read_csv(base/'split_summary.csv')
    for role,frame in [('train',train),('test',test)]:
        assert frame.source_record_id.is_unique and not frame.ml_label.isna().any() and not frame.group_id.isna().any()
        assert len(frame)==counts.loc[(counts.configuration==split)&(counts.partition==role),'rows'].item()
    assert set(train.group_id).isdisjoint(test.group_id)
    if split=='chronological_split': assert train.peak_gps_ns.max()<test.peak_gps_ns.min()
    return train,test


def restore_predictor(torch,xgb,entry,train,features,encoder,helpers):
    model_name=entry['model']
    if model_name=='XGBoost':
        model=xgb.Booster(params={'nthread':2,'device':'cpu'})
        model.load_model(entry['model_path'])
        model.set_param({'nthread':2,'device':'cpu'})
        assert model.feature_names==features
        def predict(X):
            matrix=xgb.DMatrix(X.astype(np.float32),feature_names=features,nthread=2)
            probs=model.predict(matrix)
            assert probs.shape==(len(X),24) and np.isfinite(probs).all()
            return probs.argmax(axis=1)
        return predict
    X=train[features].to_numpy(dtype=np.float64)
    y=encoder.transform(train.ml_label)
    state=read(entry['scaler'])
    assert state['features']==features
    if model_name=='MLP':
        scaler=helpers['mlp2'].restore_scaler(state,X,features)
        model=helpers['mlp2'].build_model(torch,features,entry['selected']['params'])
        model.load_state_dict(torch.load(entry['model_path'],map_location='cpu',weights_only=True))
        model.cpu().eval()
        def predict(values):
            probs,_=helpers['mlp1'].predict(torch,model,scaler,values)
            return probs.argmax(axis=1)
        return predict
    from pytorch_tabular import TabularModel
    from omegaconf import OmegaConf
    config=OmegaConf.load(entry['config'])
    assert list(config.continuous_cols)==features and config.seed==entry['seed']
    assert config.load_best is False and config.checkpoints is None
    tab=TabularModel(config=config,verbose=False)
    tab.logger=False
    frame=pd.DataFrame(X,columns=features)
    frame['ml_label']=y.astype(np.int64)
    tab.datamodule=tab.prepare_dataloader(train=frame,validation=frame.iloc[:512].copy(),seed=entry['seed'])
    tab.model=tab.prepare_model(tab.datamodule)
    tab.model.load_state_dict(torch.load(entry['model_path'],map_location='cpu',weights_only=True))
    tab.model.cpu().eval();tab._is_fitted=True
    assert tab.datamodule.label_encoder[0].classes_.tolist()==list(range(24))
    assert int(tab.datamodule.scaler.n_samples_seen_)==len(train)==state['n_samples_seen']
    for key,attr in [('mean','mean_'),('scale','scale_'),('var','var_')]:
        np.testing.assert_allclose(getattr(tab.datamodule.scaler,attr),state[key],rtol=1e-12,atol=0)
    def predict(values):
        probs,_=helpers['ft1'].predict(torch,tab,tab.datamodule.scaler,values)
        return probs.argmax(axis=1)
    return predict


def summarize(out,rows,baselines):
    raw=pd.DataFrame(rows)
    assert len(raw)==5400 and not raw.duplicated(['model','condition','seed','feature','repeat','metric']).any()
    raw.to_csv(out/'permutation_repeats.csv',index=False)
    pd.DataFrame(baselines).to_csv(out/'baseline_summary.csv',index=False)
    keys=['model','condition','split','feature_set','seed','feature','metric']
    per_seed=raw.groupby(keys,sort=False).importance.agg(['mean','std','count']).reset_index()
    per_seed=per_seed.rename(columns={'mean':'importance_mean','std':'permutation_sd','count':'permutation_repeats'})
    assert per_seed.permutation_repeats.eq(REPEATS).all()
    per_seed.to_csv(out/'permutation_importance_per_seed.csv',index=False)
    keys=[k for k in keys if k!='seed']
    overall=per_seed.groupby(keys,sort=False).importance_mean.agg(['mean','std','count']).reset_index()
    overall=overall.rename(columns={'mean':'importance_mean','std':'training_seed_sd','count':'training_seeds'})
    assert overall.training_seeds.eq(3).all()
    overall.to_csv(out/'permutation_importance_summary.csv',index=False)
    correlations=[]
    for (condition,metric),group in per_seed.groupby(['condition','metric'],sort=False):
        for seed in SEEDS+['mean_of_seed_importances']:
            subset=group if isinstance(seed,str) else group[group.seed==seed]
            vectors=subset.pivot_table(index='feature',columns='model',values='importance_mean',aggfunc='mean')
            assert vectors.shape[1]==3 and not vectors.isna().any().any()
            for a,b in itertools.combinations(MODELS,2):
                rho=rank_correlation(vectors[a].to_numpy(),vectors[b].to_numpy())
                correlations.append(dict(condition=condition,metric=metric,seed=seed,model_a=a,model_b=b,
                    n_features=len(vectors),spearman_rho=rho,
                    status='defined' if rho is not None else 'undefined_constant_vector'))
    pd.DataFrame(correlations).to_csv(out/'rank_agreement.csv',index=False)


def main():
    import torch
    import xgboost as xgb
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,required=True)
    args=ap.parse_args()
    root=args.root.expanduser().resolve()
    base=root/'jpco_o3_splits_v1';out=root/'permutation_importance_v1'
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    versions=dict(python=platform.python_version(),torch=str(torch.__version__),
        numpy=np.__version__,pandas=pd.__version__,sklearn=version('scikit-learn'),
        xgboost=xgb.__version__,pytorch_tabular=version('pytorch-tabular'),
        pytorch_lightning=version('pytorch-lightning'),torchmetrics=version('torchmetrics'))
    print('Locating reviewed runs and verifying saved artifacts...',flush=True)
    found=discover(root);helpers=import_helpers(root)
    entries,artifacts,inputs=preflight(found,versions)
    for filename,expected in inputs.items():
        assert sha(base/filename)==expected,f'Frozen data changed: {filename}'
    manifest=read(base/'manifest.json')
    protocol=dict(version='1.0',stage='common marginal permutation importance; no retraining or tuning',
        script_sha256=sha(Path(__file__)),helper_sha256=SOURCE_HASHES,versions=versions,
        reviewed_result_sha256=REVIEWED,input_sha256=inputs,artifact_sha256=artifacts,
        models=MODELS,seeds=SEEDS,feature_sets={f:manifest['feature_sets'][f] for f in FEATURE_SETS},
        selected_groups_per_split=GROUPS,cohort_sampling='5000 lowest SHA256-ranked test group IDs; all rows in each group; source order retained; labels/scores unused',
        cohort_hash_prefix='jpco-pi-v1-cohort',shuffle_hash_prefix='jpco-pi-v1-shuffle',
        shuffle_rule='PCG64 seed = first 16 hex digits of SHA256(prefix|split|feature|repeat); permutation of row indices',
        permutation_repeats=REPEATS,metrics=METRICS,importance='baseline minus permuted F1 in score units; signed, not normalized',
        repeat_variation='SD across 10 shuffles on one fixed cohort; not a sampling CI',
        seed_variation='SD across three per-seed mean importances; not a CI',
        correlation='Spearman via average ranks; constant vectors undefined; no p values',
        scope='three revised pipelines, four conditions, three training seeds',
        limitations=['marginal shuffles break correlations and can create implausible combinations',
                     'one sampled cohort per split; rare classes may be absent',
                     'does not alone establish comparability of heterogeneous explainers'],
        threads=2,device='cpu',training_performed=False)
    out.mkdir(parents=True,exist_ok=True)
    lock=out/'protocol.json'
    if lock.exists():
        assert read(lock)==protocol,'Existing permutation protocol differs. Do not mix results.'
    else:
        assert not any(out.iterdir()),'Output folder contains files without a protocol.'
        save(lock,protocol)
    lock_hash=sha(lock)
    rows=[];baselines=[];cohort_counts=[]
    for split in SPLITS:
        train,test=load_partitions(base,split,manifest['feature_sets']['original_nine'])
        encoder=LabelEncoder().fit(train.ml_label)
        assert len(encoder.classes_)==24 and set(test.ml_label)<=set(encoder.classes_)
        y_all=encoder.transform(test.ml_label)
        meta=test[['source_record_id','group_id','peak_gps_ns','ml_label']].reset_index(drop=True)
        ix=cohort_indices(test,split);cohort=test.iloc[ix].copy();y=y_all[ix]
        cohort_dir=out/'cohorts';cohort_dir.mkdir(exist_ok=True)
        cp=cohort_dir/f'{split}.csv.gz'
        cm=meta.iloc[ix].reset_index(drop=True)
        if cp.exists(): pd.testing.assert_frame_equal(pd.read_csv(cp),cm,check_dtype=False)
        else: cm.to_csv(cp,index=False,compression='gzip')
        cohort_hash=sha(cp)
        for label in encoder.classes_:
            cohort_counts.append(dict(split=split,label=label,test_support=int((test.ml_label==label).sum()),
                cohort_support=int((cohort.ml_label==label).sum()),cohort_rows=len(cohort),cohort_groups=GROUPS))
        pd.DataFrame(cohort_counts).to_csv(out/'cohort_class_counts.csv',index=False)
        print(f'\n{split}: {len(cohort):,} rows from {GROUPS:,} whole groups; {cohort.ml_label.nunique()}/24 supported classes.',flush=True)
        for featureset in FEATURE_SETS:
            features=manifest['feature_sets'][featureset];tag=f'{split}__{featureset}'
            X_all=test[features].to_numpy(dtype=np.float64)
            X=cohort[features].to_numpy(dtype=np.float64)
            assert np.isfinite(X_all).all()
            for model_name in MODELS:
                for seed in SEEDS:
                    key=f'{model_name}__{tag}__seed_{seed}'
                    entry=entries[key]
                    folder=out/'checkpoints'/key;folder.mkdir(parents=True,exist_ok=True)
                    assert read(entry['labels'])==encoder.classes_.tolist()
                    pd.testing.assert_frame_equal(pd.read_csv(entry['metadata']),meta,check_dtype=False)
                    with np.load(entry['predictions'],allow_pickle=False) as saved:
                        np.testing.assert_array_equal(saved['true_label'],y_all)
                        saved_predictions=saved['prediction'].copy()
                        if 'source_record_id' in saved:
                            np.testing.assert_array_equal(saved['source_record_id'],meta.source_record_id.astype(str).to_numpy())
                    baseline_path=folder/'baseline.json'
                    paths={f:folder/f'{f}.json' for f in features}
                    complete=baseline_path.exists() and all(p.exists() for p in paths.values())
                    predict=None
                    if complete:
                        baseline=read(baseline_path)
                        assert baseline['protocol_sha256']==lock_hash and baseline['cohort_sha256']==cohort_hash
                        assert baseline['restored_full_test_predictions_verified']
                        print(f'{key}: using completed results.',flush=True)
                    else:
                        print(f'\n{key}: restoring and checking full test predictions...',flush=True)
                        predict=restore_predictor(torch,xgb,entry,train,features,encoder,helpers)
                        t=time.perf_counter();restored=predict(X_all)
                        np.testing.assert_array_equal(restored,saved_predictions)
                        for metric,value in scores(y_all,restored).items():
                            assert np.isclose(value,entry['test_metrics']['test_'+metric],rtol=0,atol=1e-12)
                        base_pred=predict(X)
                        np.testing.assert_array_equal(base_pred,saved_predictions[ix])
                        baseline=dict(model=model_name,condition=tag,seed=seed,cohort_rows=len(y),
                            cohort_supported_classes=int(len(np.unique(y))),protocol_sha256=lock_hash,
                            cohort_sha256=cohort_hash,model_sha256=entry['model_sha256'],
                            restored_full_test_predictions_verified=True,
                            verification_seconds=time.perf_counter()-t,**scores(y,base_pred))
                        save(baseline_path,baseline)
                    baselines.append(baseline)
                    for feature in features:
                        path=paths[feature]
                        if path.exists():
                            result=read(path)
                            assert result['protocol_sha256']==lock_hash and result['cohort_sha256']==cohort_hash
                            assert result['model_sha256']==entry['model_sha256'] and result['feature']==feature
                            assert result['seed']==seed and result['model']==model_name and result['condition']==tag
                            assert len(result['rows'])==REPEATS*len(METRICS)
                        else:
                            j=features.index(feature);records=[];t=time.perf_counter()
                            for repeat in range(REPEATS):
                                perm=permutation_indices(len(y),split,feature,repeat)
                                changed=X.copy();changed[:,j]=X[perm,j]
                                measured=scores(y,predict(changed))
                                for metric in METRICS:
                                    records.append(dict(model=model_name,condition=tag,split=split,
                                        feature_set=featureset,seed=seed,feature=feature,repeat=repeat,
                                        metric=metric,baseline_score=baseline[metric],permuted_score=measured[metric],
                                        importance=baseline[metric]-measured[metric]))
                            result=dict(protocol_sha256=lock_hash,cohort_sha256=cohort_hash,
                                model_sha256=entry['model_sha256'],model=model_name,condition=tag,
                                seed=seed,feature=feature,rows=records,seconds=time.perf_counter()-t)
                            save(path,result)
                            print(f'  {feature}: 10 shuffles complete; {result["seconds"]:.1f}s',flush=True)
                        for row in result['rows']:
                            assert np.isclose(row['importance'],row['baseline_score']-row['permuted_score'],rtol=0,atol=1e-12)
                        rows.extend(result['rows'])
                    del predict,saved_predictions
                    gc.collect()
        del train,test,cohort,X,X_all,meta
        gc.collect()
    summarize(out,rows,baselines)
    print('\nPERMUTATION IMPORTANCE COMPLETE. No training or tuning performed.',flush=True)
    print(f'Outputs: {out}',flush=True)
    print('Share protocol.json, baseline_summary.csv, permutation_importance_summary.csv, permutation_importance_per_seed.csv, rank_agreement.csv, and cohort_class_counts.csv.',flush=True)


if __name__=='__main__':
    main()
