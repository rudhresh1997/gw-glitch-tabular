"""JPCO native-attribution numerical/runtime pilot, v1.0.
Run in a fresh env1 process beside common_permutation_importance.py and its six
unchanged helper scripts. No training, tuning or writes to previous results.

12 saved seed-42 models: 3 pipelines x 4 conditions. Explain 32 label-blind
whole groups from each existing PI cohort. This small sample is for checking
code, convergence and runtime, not publishing feature rankings. The planned
full comparison must use the same full cohort as permutation importance.

Output target: true-class raw score minus mean raw score over 24 classes.
This removes a class-common additive score offset. TreeSHAP is exact,
tree_path_dependent with no explicit background. Neural IG uses train-mean
and coordinatewise train-median baselines, 32/64/128 Gauss-Legendre steps.
Attributions are computed in standardized input coordinates for the neural
models (float64 scaling before float32); eval-mode embedding/BN stay inside
FT forward. Affine standardization preserves path IG with mapped baselines.

SHAP/IG explain scores, whereas PI measures F1 degradation. Matching cohorts
and target conventions does not make these definitions interchangeable.
References: https://captum.ai/api/integrated_gradients.html
https://shap.readthedocs.io/en/stable/generated/shap.TreeExplainer.html
"""
import argparse, gc, hashlib, importlib.util, json, os, platform, time, traceback
from datetime import datetime, timezone
from pathlib import Path
from importlib.metadata import version
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
          'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[k]='2'
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
PI_SCRIPT_SHA='031b03e548e65fdaa04be7686c05552ca1124d54903340c4517838e2439c2174'
PI_PROTOCOL_SHA='e6f10e69d65df629c1a4de871417c228ac63e9404d3c3eeb5c667b0dbddd60ef'
STEPS=[32,64,128]
PILOT_GROUPS=32


def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def select_pilot(cohort,split):
    key=lambda g:hashlib.sha256(f'jpco-native-pilot-v1|{split}|{g}'.encode()).hexdigest()
    groups=sorted(cohort.group_id.astype(str).unique(),key=key)[:PILOT_GROUPS]
    assert len(groups)==PILOT_GROUPS
    return np.flatnonzero(cohort.group_id.astype(str).isin(groups).to_numpy())


def restore(torch,xgb,pi,e,train,features,enc,helpers):
    """Return a raw-input probability callable and differentiable NN objects."""
    if e['model']=='XGBoost':
        net=xgb.Booster(params={'nthread':2,'device':'cpu'})
        net.load_model(e['model_path']);net.set_param({'nthread':2,'device':'cpu'})
        assert net.feature_names==features
        def probs(x):
            return net.predict(xgb.DMatrix(x.astype(np.float32),feature_names=features,nthread=2))
        return probs,net,None,None,None
    xtrain=train[features].to_numpy(dtype=np.float64)
    state=pi.read(e['scaler'])
    if e['model']=='MLP':
        scaler=helpers['mlp2'].restore_scaler(state,xtrain,features)
        net=helpers['mlp2'].build_model(torch,features,e['selected']['params'])
        net.load_state_dict(torch.load(e['model_path'],map_location='cpu',weights_only=True))
        raw_forward=net
        tab=None
    else:
        from pytorch_tabular import TabularModel
        from omegaconf import OmegaConf
        config=OmegaConf.load(e['config'])
        assert list(config.continuous_cols)==features and not config.categorical_cols
        assert config.normalize_continuous_features and config.continuous_feature_transform is None
        assert config.seed==e['seed'] and config.load_best is False and config.checkpoints is None
        tab=TabularModel(config=config,verbose=False);tab.logger=False
        frame=pd.DataFrame(xtrain,columns=features);frame['ml_label']=enc.transform(train.ml_label).astype(np.int64)
        tab.datamodule=tab.prepare_dataloader(train=frame,validation=frame.iloc[:512].copy(),seed=e['seed'])
        tab.model=tab.prepare_model(tab.datamodule)
        tab.model.load_state_dict(torch.load(e['model_path'],map_location='cpu',weights_only=True))
        tab._is_fitted=True;net=tab.model;scaler=tab.datamodule.scaler
        assert tab.datamodule.label_encoder[0].classes_.tolist()==list(range(24))
        assert state['features']==features and int(scaler.n_samples_seen_)==len(train)==state['n_samples_seen']
        for k,a in [('mean','mean_'),('scale','scale_'),('var','var_')]:
            np.testing.assert_allclose(getattr(scaler,a),state[k],rtol=1e-12,atol=0)
        def raw_forward(z):
            return net({'continuous':z,'categorical':torch.empty((len(z),0),dtype=torch.long,device=z.device)})['logits']
    net.cpu().eval()
    for parameter in net.parameters():parameter.requires_grad_(False)
    def forward(z):
        logits=raw_forward(z)
        return logits-logits.mean(dim=1,keepdim=True)
    def probs(x):
        z=torch.from_numpy(scaler.transform(x).astype(np.float32))
        with torch.no_grad():
            result=np.concatenate([raw_forward(b).softmax(dim=1).numpy() for b in z.split(512)])
        assert result.shape==(len(x),24) and np.isfinite(result).all()
        return result
    return probs,net,scaler,forward,tab


def tree_attr(shap,xgb,net,x,y,features,folder):
    t=time.perf_counter()
    explainer=shap.TreeExplainer(net,model_output='raw',feature_perturbation='tree_path_dependent')
    vals=explainer.shap_values(x.astype(np.float32),check_additivity=True,approximate=False)
    vals=np.stack(vals,axis=-1) if isinstance(vals,list) else np.asarray(vals)
    assert vals.shape==(len(x),len(features),24) and np.isfinite(vals).all(),vals.shape
    expected=np.asarray(explainer.expected_value)
    raw=net.predict(xgb.DMatrix(x.astype(np.float32),feature_names=features,nthread=2),output_margin=True)
    np.testing.assert_allclose(vals.sum(axis=1)+expected,raw,rtol=1e-4,atol=1e-4)
    centered=vals-vals.mean(axis=2,keepdims=True)
    attrs=centered[np.arange(len(x)),:,y]
    output=(raw-raw.mean(axis=1,keepdims=True))[np.arange(len(x)),y]
    base=(expected-expected.mean())[y]
    residual=attrs.sum(axis=1)-(output-base)
    np.testing.assert_allclose(attrs.sum(axis=1),output-base,rtol=1e-4,atol=1e-4)
    np.savez_compressed(folder/'treeshap.npz',all_raw_attributions=vals,expected_raw=expected,
        centered_target_attributions=attrs,target=y,centered_output=output,baseline_output=base,residual=residual)
    return attrs,dict(method='TreeSHAP',baseline='tree_path_dependent',steps=None,
        seconds=time.perf_counter()-t,max_abs_completeness_error=float(abs(residual).max()),
        numerical_checks_passed=True)


def ig_attr(torch,IntegratedGradients,forward,z,baseline,y,nsteps):
    ig=IntegratedGradients(forward,multiply_by_inputs=True)
    attributes=[];deltas=[];outputs=[];bases=[];start=time.perf_counter()
    # At most 128 interpolation examples per gradient call, eight input rows.
    for offset in range(0,len(z),8):
        b=z[offset:offset+8].clone().requires_grad_(True)
        target=torch.from_numpy(y[offset:offset+8]).long()
        attr,delta=ig.attribute(b,baselines=baseline,target=target,n_steps=nsteps,
            method='gausslegendre',internal_batch_size=128,return_convergence_delta=True)
        with torch.no_grad():
            f=forward(b).gather(1,target[:,None]).squeeze(1)
            f0=forward(baseline.expand(len(b),-1)).gather(1,target[:,None]).squeeze(1)
        attributes.append(attr.detach().numpy());deltas.append(delta.detach().numpy())
        outputs.append(f.numpy());bases.append(f0.numpy())
    a=np.concatenate(attributes);delta=np.concatenate(deltas);f=np.concatenate(outputs);f0=np.concatenate(bases)
    assert all(np.isfinite(v).all() for v in [a,delta,f,f0]),'Non-finite IG values'
    residual=a.sum(axis=1)-(f-f0)
    np.testing.assert_allclose(residual,delta,rtol=1e-4,atol=1e-4)
    scaled=abs(residual)/(1+abs(f-f0))
    return a,dict(delta=delta,output=f,baseline_output=f0,residual=residual),dict(
        seconds=time.perf_counter()-start,max_abs_completeness_error=float(abs(residual).max()),
        median_scaled_completeness_error=float(np.median(scaled)),
        p95_scaled_completeness_error=float(np.quantile(scaled,.95)),
        max_scaled_completeness_error=float(scaled.max()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);args=parser.parse_args()
    root=args.root.expanduser().resolve()
    import torch,xgboost as xgb,shap
    from captum.attr import IntegratedGradients
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    source=root/'common_permutation_importance.py'
    assert digest(source)==PI_SCRIPT_SHA,'Restore unchanged common_permutation_importance.py'
    spec=importlib.util.spec_from_file_location('frozen_pi',source);pi=importlib.util.module_from_spec(spec);spec.loader.exec_module(pi)
    old=root/'permutation_importance_v1';base=root/'jpco_o3_splits_v1'
    assert digest(old/'protocol.json')==PI_PROTOCOL_SHA,'Permutation protocol differs from reviewed version.'
    versions=dict(python=platform.python_version(),torch=str(torch.__version__),numpy=np.__version__,
        pandas=pd.__version__,sklearn=version('scikit-learn'),xgboost=xgb.__version__,
        pytorch_tabular=version('pytorch-tabular'),pytorch_lightning=version('pytorch-lightning'),
        torchmetrics=version('torchmetrics'),shap=version('shap'),captum=version('captum'))
    assert versions['shap']=='0.47.2' and versions['captum']=='0.7.0','Use the reported env1 versions.'
    print('Verifying reviewed scripts, models, preprocessing and data...',flush=True)
    helpers=pi.import_helpers(root);found=pi.discover(root);entries,artifacts,inputs=pi.preflight(found,versions)
    for filename,h in inputs.items():assert digest(base/filename)==h,f'Data changed: {filename}'
    prior=pi.read(old/'protocol.json');assert prior['artifact_sha256']==artifacts
    prior_baselines=pd.read_csv(old/'baseline_summary.csv')
    assert prior_baselines.protocol_sha256.eq(PI_PROTOCOL_SHA).all()
    manifest=pi.read(base/'manifest.json')
    out=root/'native_attribution_pilots'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out.mkdir(parents=True,exist_ok=False)
    protocol=dict(version='1.0',pilot_only=True,script_sha256=digest(Path(__file__)),versions=versions,
        permutation_protocol_sha256=PI_PROTOCOL_SHA,permutation_script_sha256=PI_SCRIPT_SHA,
        artifact_sha256=artifacts,input_sha256=inputs,models=pi.MODELS,seed=42,groups_per_split=PILOT_GROUPS,
        cohort_rule='Lowest SHA256(jpco-native-pilot-v1|split|group) within existing PI cohort; whole groups, source order',
        target='true-class raw score minus mean of all 24 raw class scores',
        treeshap=dict(feature_perturbation='tree_path_dependent',model_output='raw',approximate=False),
        ig=dict(baselines=['training_mean','training_median'],steps=STEPS,method='gausslegendre',
                internal_batch_size=128,input_batch_size=8,multiply_by_inputs=True),
        diagnostic_gate='128-step p95 abs(delta)/(1+abs(output-baseline)) <= 0.001 AND 64->128 mean-absolute feature-vector relative L1 <= 0.02',
        gate_note='Practical pilot diagnostic, not a guarantee of exact attribution or scientific validity.',
        limitations=['32 groups, seed42 only; no final rankings or comparisons to full-cohort PI',
                     'SHAP/IG score attribution and F1 permutation degradation are different quantities',
                     'synthetic mean/median references and marginal feature operations can be off manifold'],
        training_performed=False,device='cpu',threads=2)
    pi.save(out/'pilot_protocol.json',protocol)
    diagnostics=[];features_out=[];verifications=[];cohort_counts=[]
    start=time.perf_counter()
    try:
        for split in pi.SPLITS:
            train,test=pi.load_partitions(base,split,manifest['feature_sets']['original_nine'])
            enc=LabelEncoder().fit(train.ml_label);assert len(enc.classes_)==24
            yall=enc.transform(test.ml_label)
            meta=test[['source_record_id','group_id','peak_gps_ns','ml_label']].reset_index(drop=True)
            cohort_ix=pi.cohort_indices(test,split);cohort=test.iloc[cohort_ix]
            saved_cohort=old/'cohorts'/f'{split}.csv.gz'
            pd.testing.assert_frame_equal(pd.read_csv(saved_cohort),meta.iloc[cohort_ix].reset_index(drop=True),check_dtype=False)
            pilot_ix=select_pilot(cohort,split);ix=cohort_ix[pilot_ix];y=yall[ix]
            pilot_meta=meta.iloc[ix].reset_index(drop=True);pilot_meta.to_csv(out/f'{split}_pilot_records.csv',index=False)
            cb=prior_baselines[prior_baselines.condition.str.startswith(split+'__')]
            assert cb.cohort_sha256.eq(digest(saved_cohort)).all() and cb.restored_full_test_predictions_verified.eq(True).all()
            for label in enc.classes_:
                cohort_counts.append(dict(split=split,label=label,pilot_support=int((pilot_meta.ml_label==label).sum())))
            print(f'\n{split}: {len(ix)} rows in {PILOT_GROUPS} groups; {len(np.unique(y))}/24 classes.',flush=True)
            for feature_set in pi.FEATURE_SETS:
                features=manifest['feature_sets'][feature_set];tag=f'{split}__{feature_set}'
                xall=test[features].to_numpy(dtype=np.float64);x=xall[ix]
                for model in pi.MODELS:
                    key=f'{model}__{tag}__seed_42';e=entries[key];folder=out/key;folder.mkdir()
                    print(f'\n{key}: restoring model...',flush=True);t=time.perf_counter()
                    assert pi.read(e['labels'])==enc.classes_.tolist()
                    pd.testing.assert_frame_equal(pd.read_csv(e['metadata']),meta,check_dtype=False)
                    with np.load(e['predictions'],allow_pickle=False) as f:
                        np.testing.assert_array_equal(f['true_label'],yall);saved_pred=f['prediction'].copy()
                    probs,net,scaler,forward,tab=restore(torch,xgb,pi,e,train,features,enc,helpers)
                    predicted=probs(xall).argmax(axis=1)
                    np.testing.assert_array_equal(predicted,saved_pred)
                    px=probs(x);np.testing.assert_array_equal(px.argmax(axis=1),saved_pred[ix])
                    probability_error=0.0
                    if model=='FT-Transformer':
                        api_probs,_=helpers['ft1'].predict(torch,tab,scaler,x)
                        probability_error=float(abs(api_probs-px).max())
                        np.testing.assert_allclose(api_probs,px,rtol=1e-5,atol=1e-6)
                    verifications.append(dict(model=model,condition=tag,seed=42,pilot_rows=len(x),
                        model_sha256=e['model_sha256'],full_test_prediction_replay_passed=True,
                        ft_api_probability_max_error=probability_error,seconds=time.perf_counter()-t))
                    if model=='XGBoost':
                        a,diag=tree_attr(shap,xgb,net,x,y,features,folder)
                        diagnostics.append(dict(model=model,condition=tag,**diag))
                        for feature,value in zip(features,abs(a).mean(axis=0)):
                            features_out.append(dict(model=model,condition=tag,baseline='tree_path_dependent',steps=0,feature=feature,mean_abs_attribution=float(value)))
                        print(f'  TreeSHAP: {diag["seconds"]:.1f}s; additivity checked.',flush=True)
                    else:
                        z=torch.from_numpy(scaler.transform(x).astype(np.float32))
                        refs={'training_mean':np.asarray(scaler.mean_,dtype=np.float64),
                              'training_median':np.median(train[features].to_numpy(dtype=np.float64),axis=0)}
                        for baseline_name,raw_baseline in refs.items():
                            bz=torch.from_numpy(scaler.transform(raw_baseline[None,:]).astype(np.float32))
                            previous=None
                            for steps in STEPS:
                                a,detail,diag=ig_attr(torch,IntegratedGradients,forward,z,bz,y,steps)
                                mean_abs=abs(a).mean(axis=0)
                                rel_change=None if previous is None else float(abs(mean_abs-previous).sum()/max(abs(mean_abs).sum(),1e-12))
                                passed=None if steps!=128 else bool(diag['p95_scaled_completeness_error']<=.001 and rel_change<=.02)
                                diagnostics.append(dict(model=model,condition=tag,method='IntegratedGradients',baseline=baseline_name,
                                    steps=steps,relative_L1_change_from_previous_steps=rel_change,numerical_checks_passed=passed,**diag))
                                np.savez_compressed(folder/f'ig_{baseline_name}_{steps}.npz',attributions=a,
                                    raw_baseline=raw_baseline,standardized_baseline=bz.numpy(),target=y,**detail)
                                for feature,value in zip(features,mean_abs):
                                    features_out.append(dict(model=model,condition=tag,baseline=baseline_name,steps=steps,feature=feature,mean_abs_attribution=float(value)))
                                print(f'  IG {baseline_name}, {steps} steps: {diag["seconds"]:.1f}s; p95 scaled error={diag["p95_scaled_completeness_error"]:.3g}; change={rel_change}',flush=True)
                                previous=mean_abs
                    pd.DataFrame(diagnostics).to_csv(out/'numerical_checks.csv',index=False)
                    pd.DataFrame(features_out).to_csv(out/'pilot_feature_summaries.csv',index=False)
                    pd.DataFrame(verifications).to_csv(out/'prediction_replay.csv',index=False)
                    del probs,net,scaler,forward,tab,px,predicted,saved_pred
                    gc.collect()
            del train,test,xall,x,cohort,meta
            gc.collect()
        pd.DataFrame(cohort_counts).to_csv(out/'pilot_class_counts.csv',index=False)
        checks=[d for d in diagnostics if d.get('steps')==128]
        assert len(checks)==16 and len(verifications)==12
        failures=[dict(model=d['model'],condition=d['condition'],baseline=d['baseline']) for d in checks if not d['numerical_checks_passed']]
        summary=dict(pilot_completed=True,training_performed=False,models_replayed=12,ig_checks_at_128_steps=16,
            all_numerical_checks_passed=not failures,needs_numerical_followup=failures,
            elapsed_seconds=time.perf_counter()-start,output_directory=str(out),
            scope='Pilot only; do not interpret as full-cohort native-versus-permutation evidence.',
            next='Share pilot_summary.json, numerical_checks.csv, prediction_replay.csv and pilot_protocol.json.')
        pi.save(out/'pilot_summary.json',summary)
        print('\nPILOT COMPLETE\n'+json.dumps(summary,indent=2),flush=True)
    except Exception:
        (out/'failure.txt').write_text(traceback.format_exc())
        print(f'Pilot stopped; diagnostic files: {out}',flush=True)
        raise


if __name__=='__main__':main()
