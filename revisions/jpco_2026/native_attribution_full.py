"""Full matched-cohort native attribution analysis for JPCO revision.

No training or tuning. Restores the 36 frozen models already evaluated by the
reviewed common-permutation analysis. Uses the same 5000-group test cohort per
split. XGBoost: exact TreeSHAP, raw margin, tree-path-dependent. MLP and
FT-Transformer: 128-step Gauss-Legendre Integrated Gradients with training-mean
and coordinatewise training-median baselines. Target for every event is its
true-class raw score minus the mean raw score across all 24 classes.

The script resumes completed model/baseline units only under an identical
protocol. Outputs include native-vs-permutation rank comparisons, cross-model
native rank comparisons and IG baseline sensitivity. Raw-score attribution and
F1 permutation degradation remain different explanation quantities. Neural raw
score scales also differ by fitted model, so magnitudes are not compared across
architectures; rank comparisons are descriptive over only six or nine features.
"""
import argparse, gc, hashlib, importlib.util, itertools, json, os, platform, time, traceback
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[k]='2'
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

PI_SCRIPT_SHA='031b03e548e65fdaa04be7686c05552ca1124d54903340c4517838e2439c2174'
PI_PROTOCOL_SHA='e6f10e69d65df629c1a4de871417c228ac63e9404d3c3eeb5c667b0dbddd60ef'
PILOT_SCRIPT_SHA='4e8417b1e89a8f052ab1e3b3984c8e414e427c036d34d8da834dd873bf692b3d'
STEPS=128
BASELINES=['training_mean','training_median']


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def save(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)


def spearman(a,b):
    a,b=np.asarray(a,float),np.asarray(b,float)
    ra=pd.Series(a).rank(method='average').to_numpy();rb=pd.Series(b).rank(method='average').to_numpy()
    if np.ptp(ra)==0 or np.ptp(rb)==0:return None
    return float(np.corrcoef(ra,rb)[0,1])


def aggregate_and_compare(out,rows,pi_per_seed):
    native=pd.DataFrame(rows).sort_values(['model','condition','seed','baseline','feature']).reset_index(drop=True)
    native.to_csv(out/'native_importance_per_seed.csv',index=False)
    keys=['model','condition','split','feature_set','baseline','feature']
    summary=(native.groupby(keys,sort=False).mean_abs_attribution
             .agg(importance_mean='mean',training_seed_sd='std',training_seeds='count').reset_index())
    assert summary.training_seeds.eq(3).all()
    summary.to_csv(out/'native_importance_summary.csv',index=False)
    pi_per_seed=pi_per_seed.copy();pi_per_seed.seed=pi_per_seed.seed.astype(int)
    within=[]
    for key,g in native.groupby(['model','condition','split','feature_set','seed','baseline'],sort=False):
        model,condition,split,feature_set,seed,baseline=key
        for metric,q in pi_per_seed[(pi_per_seed.model==model)&(pi_per_seed.condition==condition)&(pi_per_seed.seed==seed)].groupby('metric'):
            z=g[['feature','mean_abs_attribution']].merge(q[['feature','importance_mean']],on='feature',validate='one_to_one')
            assert len(z)==len(g)==len(q)
            rho=spearman(z.mean_abs_attribution,z.importance_mean)
            within.append(dict(model=model,condition=condition,split=split,feature_set=feature_set,seed=seed,
                native_baseline=baseline,permutation_metric=metric,n_features=len(z),spearman_rho=rho,status='defined' if rho is not None else 'undefined_constant_vector'))
    # Correlation of the independently seed-averaged vectors.
    for key,g in summary.groupby(['model','condition','split','feature_set','baseline'],sort=False):
        model,condition,split,feature_set,baseline=key
        q0=pi_per_seed[(pi_per_seed.model==model)&(pi_per_seed.condition==condition)]
        for metric,q in q0.groupby('metric'):
            q=q.groupby('feature',as_index=False).importance_mean.mean()
            z=g[['feature','importance_mean']].merge(q,on='feature',suffixes=('_native','_permutation'),validate='one_to_one')
            rho=spearman(z.importance_mean_native,z.importance_mean_permutation)
            within.append(dict(model=model,condition=condition,split=split,feature_set=feature_set,seed='mean_of_seed_importances',
                native_baseline=baseline,permutation_metric=metric,n_features=len(z),spearman_rho=rho,status='defined' if rho is not None else 'undefined_constant_vector'))
    pd.DataFrame(within).to_csv(out/'native_vs_permutation_rank_agreement.csv',index=False)

    # Add averaged vectors as a fourth seed label for cross-model comparisons.
    avg=summary.rename(columns={'importance_mean':'mean_abs_attribution'})
    avg['seed']='mean_of_seed_importances'
    combined=pd.concat([native,avg[native.columns]],ignore_index=True)
    cross=[]
    for condition in sorted(native.condition.unique()):
        for seed in [42,43,44,'mean_of_seed_importances']:
            for baseline in BASELINES:
                vectors={}
                for model in ['XGBoost','MLP','FT-Transformer']:
                    b='tree_path_dependent' if model=='XGBoost' else baseline
                    q=combined[(combined.model==model)&(combined.condition==condition)&(combined.seed.astype(str)==str(seed))&(combined.baseline==b)]
                    assert len(q) in [6,9]
                    vectors[model]=q.set_index('feature').mean_abs_attribution
                for a,b in itertools.combinations(vectors,2):
                    z=pd.concat([vectors[a].rename(a),vectors[b].rename(b)],axis=1,join='inner')
                    assert len(z)==len(vectors[a])==len(vectors[b])
                    rho=spearman(z[a],z[b])
                    cross.append(dict(condition=condition,seed=seed,neural_ig_baseline=baseline,model_a=a,model_b=b,
                        n_features=len(z),spearman_rho=rho,status='defined' if rho is not None else 'undefined_constant_vector'))
    pd.DataFrame(cross).to_csv(out/'native_cross_model_rank_agreement.csv',index=False)

    sensitivity=[]
    for key,g in combined[combined.model.isin(['MLP','FT-Transformer'])].groupby(['model','condition','seed'],sort=False):
        model,condition,seed=key
        z=g.pivot(index='feature',columns='baseline',values='mean_abs_attribution')
        assert set(z.columns)==set(BASELINES)
        rho=spearman(z.training_mean,z.training_median)
        sensitivity.append(dict(model=model,condition=condition,seed=seed,n_features=len(z),spearman_rho=rho,
            status='defined' if rho is not None else 'undefined_constant_vector'))
    pd.DataFrame(sensitivity).to_csv(out/'ig_baseline_rank_sensitivity.csv',index=False)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,required=True);args=ap.parse_args()
    root=args.root.expanduser().resolve();base=root/'jpco_o3_splits_v1';prior=root/'permutation_importance_v1';out=root/'native_attribution_v1'
    import torch,xgboost as xgb,shap
    from captum.attr import IntegratedGradients
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    pi_path=root/'common_permutation_importance.py';pilot_path=root/'native_attribution_pilot.py'
    assert sha(pi_path)==PI_SCRIPT_SHA,'Restore unchanged common_permutation_importance.py'
    assert sha(pilot_path)==PILOT_SCRIPT_SHA,'Restore unchanged native_attribution_pilot.py'
    def load_module(name,path):
        spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
    pi=load_module('frozen_pi',pi_path);pilot=load_module('frozen_native_pilot',pilot_path)
    assert sha(prior/'protocol.json')==PI_PROTOCOL_SHA,'Permutation protocol differs from reviewed version.'
    versions=dict(python=platform.python_version(),torch=str(torch.__version__),numpy=np.__version__,pandas=pd.__version__,
        sklearn=version('scikit-learn'),xgboost=xgb.__version__,pytorch_tabular=version('pytorch-tabular'),
        pytorch_lightning=version('pytorch-lightning'),torchmetrics=version('torchmetrics'),shap=version('shap'),captum=version('captum'))
    assert versions['shap']=='0.47.2' and versions['captum']=='0.7.0'
    print('Verifying frozen inputs, scripts, models and reviewed permutation results...',flush=True)
    helpers=pi.import_helpers(root);found=pi.discover(root);entries,artifacts,inputs=pi.preflight(found,versions)
    for f,h in inputs.items():assert sha(base/f)==h,f'Frozen input changed: {f}'
    prior_protocol=pi.read(prior/'protocol.json');assert prior_protocol['artifact_sha256']==artifacts
    pi_seed_path=prior/'permutation_importance_per_seed.csv';assert pi_seed_path.is_file()
    pi_per_seed=pd.read_csv(pi_seed_path);assert len(pi_per_seed)==540
    protocol=dict(version='1.0',stage='full matched-cohort native attribution; no training or tuning',script_sha256=sha(Path(__file__)),
        permutation_protocol_sha256=PI_PROTOCOL_SHA,permutation_script_sha256=PI_SCRIPT_SHA,pilot_script_sha256=PILOT_SCRIPT_SHA,
        versions=versions,artifact_sha256=artifacts,input_sha256=inputs,models=pi.MODELS,seeds=pi.SEEDS,
        feature_sets=prior_protocol['feature_sets'],cohort_source='exact saved 5000-group permutation cohort per split',
        target='true-class raw score minus mean raw score over all 24 classes',
        treeshap=dict(feature_perturbation='tree_path_dependent',model_output='raw',approximate=False,additivity_check=True),
        integrated_gradients=dict(baselines=BASELINES,steps=STEPS,method='gausslegendre',internal_batch_size=128,input_batch_size=8,multiply_by_inputs=True),
        aggregation='mean absolute attribution by feature within cohort; then mean and sample SD across three training seeds',
        comparisons=['native versus permutation ranks per model','native cross-model ranks','IG mean-versus-median baseline ranks'],
        correlation='Spearman via average ranks; constant vectors undefined; no p values or confidence intervals',
        numerical_gate='IG p95 abs(completeness residual)/(1+abs(output-baseline)) <= 0.001; TreeSHAP additivity rtol/atol 1e-4',
        limitations=['score attribution and F1 permutation degradation are different quantities','raw score scaling differs across fitted models',
                     'six/nine features and three seeds; correlations are descriptive','synthetic mean/median baselines and TreeSHAP dependence assumption'],
        training_performed=False,device='cpu',threads=2)
    out.mkdir(parents=True,exist_ok=True);lock=out/'protocol.json'
    if lock.exists():assert pi.read(lock)==protocol,'Existing output protocol differs. Do not mix results.'
    else:
        assert not any(out.iterdir()),'Output folder contains files without a protocol.'
        save(lock,protocol)
    lock_hash=sha(lock);rows=[];diagnostics=[];replay=[];start=time.perf_counter()
    manifest=pi.read(base/'manifest.json')
    try:
        for split in pi.SPLITS:
            train,test=pi.load_partitions(base,split,manifest['feature_sets']['original_nine'])
            enc=LabelEncoder().fit(train.ml_label);yall=enc.transform(test.ml_label)
            meta=test[['source_record_id','group_id','peak_gps_ns','ml_label']].reset_index(drop=True)
            ix=pi.cohort_indices(test,split);cohort=test.iloc[ix].copy();y=yall[ix]
            saved_cohort=prior/'cohorts'/f'{split}.csv.gz'
            pd.testing.assert_frame_equal(pd.read_csv(saved_cohort),meta.iloc[ix].reset_index(drop=True),check_dtype=False)
            cohort_hash=sha(saved_cohort)
            print(f'\n{split}: exact reviewed cohort, {len(cohort):,} rows, {cohort.group_id.nunique():,} groups, {cohort.ml_label.nunique()}/24 classes.',flush=True)
            for feature_set in pi.FEATURE_SETS:
                features=manifest['feature_sets'][feature_set];tag=f'{split}__{feature_set}'
                x=cohort[features].to_numpy(dtype=np.float64)
                for model in pi.MODELS:
                    for seed in pi.SEEDS:
                        key=f'{model}__{tag}__seed_{seed}';e=entries[key];folder=out/'checkpoints'/key;folder.mkdir(parents=True,exist_ok=True)
                        expected=['tree_path_dependent'] if model=='XGBoost' else BASELINES
                        paths={b:folder/f'{b}.json' for b in expected}
                        if all(path.exists() for path in paths.values()):
                            results=[pi.read(paths[b]) for b in expected]
                            for result,b in zip(results,expected):
                                assert result['protocol_sha256']==lock_hash and result['model_sha256']==e['model_sha256']
                                assert result['cohort_sha256']==cohort_hash and result['baseline']==b and result['condition']==tag and result['seed']==seed
                            replay.append(dict(model=model,condition=tag,seed=seed,rows=len(x),model_sha256=e['model_sha256'],
                                cohort_prediction_replay_passed=results[0]['cohort_prediction_replay_passed'],
                                ft_api_probability_max_error=results[0]['ft_api_probability_max_error'],
                                seconds=results[0]['prediction_verification_seconds']))
                            print(f'{key}: using completed results.',flush=True)
                        else:
                            assert not any(path.exists() for path in paths.values()),f'Partial unit at {folder}; delete this unit folder and resume.'
                            print(f'\n{key}: restoring and verifying cohort predictions...',flush=True);t=time.perf_counter()
                            assert pi.read(e['labels'])==enc.classes_.tolist();pd.testing.assert_frame_equal(pd.read_csv(e['metadata']),meta,check_dtype=False)
                            with np.load(e['predictions'],allow_pickle=False) as f:
                                np.testing.assert_array_equal(f['true_label'],yall);saved_pred=f['prediction'].copy()
                            probs,net,scaler,forward,tab=pilot.restore(torch,xgb,pi,e,train,features,enc,helpers)
                            px=probs(x);np.testing.assert_array_equal(px.argmax(axis=1),saved_pred[ix])
                            api_error=0.0
                            if model=='FT-Transformer':
                                api,_=helpers['ft1'].predict(torch,tab,scaler,x);api_error=float(abs(api-px).max());np.testing.assert_allclose(api,px,rtol=1e-5,atol=1e-6)
                            replay.append(dict(model=model,condition=tag,seed=seed,rows=len(x),model_sha256=e['model_sha256'],
                                cohort_prediction_replay_passed=True,ft_api_probability_max_error=api_error,seconds=time.perf_counter()-t))
                            verification=dict(cohort_prediction_replay_passed=True,ft_api_probability_max_error=api_error,
                                              prediction_verification_seconds=replay[-1]['seconds'])
                            if model=='XGBoost':
                                attrs,diag=pilot.tree_attr(shap,xgb,net,x,y,features,folder)
                                result=dict(protocol_sha256=lock_hash,cohort_sha256=cohort_hash,model_sha256=e['model_sha256'],model=model,
                                    condition=tag,split=split,feature_set=feature_set,seed=seed,
                                    feature_importance={f:float(v) for f,v in zip(features,abs(attrs).mean(axis=0))},**verification,**diag)
                                save(paths['tree_path_dependent'],result);results=[result]
                            else:
                                z=torch.from_numpy(scaler.transform(x).astype(np.float32));results=[]
                                refs={'training_mean':np.asarray(scaler.mean_,dtype=np.float64),
                                      'training_median':np.median(train[features].to_numpy(dtype=np.float64),axis=0)}
                                for baseline,raw_base in refs.items():
                                    bz=torch.from_numpy(scaler.transform(raw_base[None,:]).astype(np.float32))
                                    attrs,detail,diag=pilot.ig_attr(torch,IntegratedGradients,forward,z,bz,y,STEPS)
                                    assert diag['p95_scaled_completeness_error']<=.001,'IG numerical gate failed; share output folder.'
                                    np.savez_compressed(folder/f'{baseline}.npz',attributions=attrs,raw_baseline=raw_base,
                                        standardized_baseline=bz.numpy(),target=y,**detail)
                                    result=dict(protocol_sha256=lock_hash,cohort_sha256=cohort_hash,model_sha256=e['model_sha256'],model=model,
                                        condition=tag,split=split,feature_set=feature_set,seed=seed,baseline=baseline,steps=STEPS,
                                        feature_importance={f:float(v) for f,v in zip(features,abs(attrs).mean(axis=0))},
                                        numerical_checks_passed=True,**verification,**diag)
                                    save(paths[baseline],result);results.append(result)
                            del probs,net,scaler,forward,tab,px,saved_pred
                            gc.collect()
                        for result in results:
                            diagnostics.append({k:v for k,v in result.items() if k not in ['feature_importance','protocol_sha256','cohort_sha256','model_sha256']})
                            for feature,value in result['feature_importance'].items():
                                rows.append(dict(model=model,condition=tag,split=split,feature_set=feature_set,seed=seed,
                                    baseline=result['baseline'],feature=feature,mean_abs_attribution=value))
                        pd.DataFrame(diagnostics).to_csv(out/'numerical_checks.csv',index=False)
                        pd.DataFrame(replay).to_csv(out/'cohort_prediction_replay.csv',index=False)
            del train,test,cohort,meta
            gc.collect()
        aggregate_and_compare(out,rows,pi_per_seed)
        summary=dict(completed=True,training_performed=False,model_condition_seed_units=36,native_feature_rows=len(rows),
            numerical_units=len(diagnostics),elapsed_seconds=time.perf_counter()-start,output_directory=str(out),
            next='Share summary.json, protocol.json, native_importance_per_seed.csv, native_importance_summary.csv, native_vs_permutation_rank_agreement.csv, native_cross_model_rank_agreement.csv, ig_baseline_rank_sensitivity.csv, numerical_checks.csv, and cohort_prediction_replay.csv.')
        save(out/'summary.json',summary)
        print('\nFULL NATIVE ATTRIBUTION COMPLETE\n'+json.dumps(summary,indent=2),flush=True)
    except Exception:
        (out/'failure.txt').write_text(traceback.format_exc());print(f'Run stopped. Share failure.txt from {out}',flush=True);raise

if __name__=='__main__':main()
