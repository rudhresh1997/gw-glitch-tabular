"""Adaptive full matched-cohort native attribution analysis for JPCO revision.

Version 2 reuses every valid checkpoint from native_attribution_v1. Missing
Integrated Gradients units start at 128 Gauss-Legendre steps and retry at 256
and 512 only when the full-cohort p95 scaled completeness error exceeds 1e-3.
No model is trained or tuned. Previous outputs are read-only.
"""
import argparse, gc, hashlib, importlib.util, itertools, json, os, platform, shutil, time, traceback
from importlib.metadata import version
from pathlib import Path

for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[name]='2'

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

PI_SCRIPT_SHA='031b03e548e65fdaa04be7686c05552ca1124d54903340c4517838e2439c2174'
PI_PROTOCOL_SHA='e6f10e69d65df629c1a4de871417c228ac63e9404d3c3eeb5c667b0dbddd60ef'
PILOT_SCRIPT_SHA='4e8417b1e89a8f052ab1e3b3984c8e414e427c036d34d8da834dd873bf692b3d'
V1_SCRIPT_SHA='b3400426698d3ebdaad9304ef3fd34edba499b43bfbfa68fd22ee8b9a1cf480f'
BASELINES=['training_mean','training_median']
STEP_SCHEDULE=[128,256,512]
GATE=.001


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def save(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    os.replace(temporary,path)


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def valid_prior_result(path,old_protocol_hash,entry,cohort_hash,condition,seed,baseline):
    if not path.is_file():
        return None
    value=json.loads(path.read_text())
    required=(value.get('protocol_sha256')==old_protocol_hash and
              value.get('model_sha256')==entry['model_sha256'] and
              value.get('cohort_sha256')==cohort_hash and
              value.get('condition')==condition and value.get('seed')==seed and
              value.get('baseline')==baseline and
              value.get('cohort_prediction_replay_passed') is True and
              value.get('numerical_checks_passed') is True)
    if not required:
        return None
    if baseline!='tree_path_dependent' and value.get('p95_scaled_completeness_error',np.inf)>GATE:
        return None
    return value


def diagnostic_row(result):
    excluded={'feature_importance','protocol_sha256','cohort_sha256','model_sha256','attempts'}
    row={key:value for key,value in result.items() if key not in excluded}
    row['attempt_count']=len(result.get('attempts',[])) or 1
    return row


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();root=args.root.expanduser().resolve()
    base=root/'jpco_o3_splits_v1';permutation=root/'permutation_importance_v1'
    previous=root/'native_attribution_v1';out=root/'native_attribution_v2'

    import torch,xgboost as xgb,shap
    from captum.attr import IntegratedGradients
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)

    paths={'pi':root/'common_permutation_importance.py','pilot':root/'native_attribution_pilot.py',
           'v1_script':root/'native_attribution_full.py'}
    assert sha(paths['pi'])==PI_SCRIPT_SHA,'Restore unchanged common_permutation_importance.py'
    assert sha(paths['pilot'])==PILOT_SCRIPT_SHA,'Restore unchanged native_attribution_pilot.py'
    assert sha(paths['v1_script'])==V1_SCRIPT_SHA,'Restore the unchanged version-1 full script.'
    assert sha(permutation/'protocol.json')==PI_PROTOCOL_SHA,'Reviewed permutation protocol changed.'
    pi=load_module('frozen_pi',paths['pi']);pilot=load_module('frozen_pilot',paths['pilot'])
    v1=load_module('frozen_v1',paths['v1_script'])

    versions=dict(python=platform.python_version(),torch=str(torch.__version__),numpy=np.__version__,
        pandas=pd.__version__,sklearn=version('scikit-learn'),xgboost=xgb.__version__,
        pytorch_tabular=version('pytorch-tabular'),pytorch_lightning=version('pytorch-lightning'),
        torchmetrics=version('torchmetrics'),shap=version('shap'),captum=version('captum'))
    assert versions['shap']=='0.47.2' and versions['captum']=='0.7.0'

    print('Verifying frozen inputs, models, permutation results and reusable version-1 checkpoints...',flush=True)
    helpers=pi.import_helpers(root);found=pi.discover(root)
    entries,artifacts,inputs=pi.preflight(found,versions)
    for filename,expected in inputs.items():
        assert sha(base/filename)==expected,f'Frozen input changed: {filename}'
    permutation_protocol=pi.read(permutation/'protocol.json')
    assert permutation_protocol['artifact_sha256']==artifacts
    pi_per_seed=pd.read_csv(permutation/'permutation_importance_per_seed.csv')
    assert len(pi_per_seed)==540

    old_protocol=None;old_protocol_hash=None
    if previous.joinpath('protocol.json').is_file():
        old_protocol=pi.read(previous/'protocol.json');old_protocol_hash=sha(previous/'protocol.json')
        assert old_protocol['script_sha256']==V1_SCRIPT_SHA
        assert old_protocol['permutation_protocol_sha256']==PI_PROTOCOL_SHA
        assert old_protocol['artifact_sha256']==artifacts and old_protocol['input_sha256']==inputs

    protocol=dict(version='2.0',stage='adaptive full matched-cohort native attribution; no training or tuning',
        script_sha256=sha(Path(__file__)),source_script_sha256={'permutation':PI_SCRIPT_SHA,'pilot':PILOT_SCRIPT_SHA,'v1':V1_SCRIPT_SHA},
        permutation_protocol_sha256=PI_PROTOCOL_SHA,versions=versions,artifact_sha256=artifacts,input_sha256=inputs,
        models=pi.MODELS,seeds=pi.SEEDS,feature_sets=permutation_protocol['feature_sets'],
        cohort_source='exact saved 5000-group permutation cohort per split',
        target='true-class raw score minus mean raw score over all 24 classes',
        treeshap=dict(feature_perturbation='tree_path_dependent',model_output='raw',approximate=False,additivity_check=True),
        integrated_gradients=dict(baselines=BASELINES,step_schedule=STEP_SCHEDULE,method='gausslegendre',
            internal_batch_size=128,input_batch_size=8,multiply_by_inputs=True,
            acceptance=f'p95 abs(completeness residual)/(1+abs(output-baseline)) <= {GATE}'),
        checkpoint_reuse='valid version-1 results are copied with provenance; failed or absent units are recomputed adaptively',
        aggregation='mean absolute attribution within cohort; mean and sample SD across three training seeds',
        comparisons=['native versus permutation ranks per model','native cross-model ranks','IG mean-versus-median baseline ranks'],
        limitations=['score attribution and F1 permutation degradation are different quantities','raw score scaling differs across models',
                     'six/nine features and three seeds; correlations are descriptive','synthetic baselines and TreeSHAP dependence assumption'],
        training_performed=False,device='cpu',threads=2)
    out.mkdir(parents=True,exist_ok=True);lock=out/'protocol.json'
    if lock.exists():
        assert pi.read(lock)==protocol,'Existing version-2 protocol differs. Do not mix results.'
    else:
        assert not any(out.iterdir()),'Version-2 output folder contains files without a protocol.'
        save(lock,protocol)
    lock_hash=sha(lock);manifest=pi.read(base/'manifest.json')
    rows=[];diagnostics=[];replay=[];start=time.perf_counter();imported=0;computed=0

    try:
        for split in pi.SPLITS:
            train,test=pi.load_partitions(base,split,manifest['feature_sets']['original_nine'])
            encoder=LabelEncoder().fit(train.ml_label);yall=encoder.transform(test.ml_label)
            metadata=test[['source_record_id','group_id','peak_gps_ns','ml_label']].reset_index(drop=True)
            indices=pi.cohort_indices(test,split);cohort=test.iloc[indices].copy();y=yall[indices]
            saved_cohort=permutation/'cohorts'/f'{split}.csv.gz'
            pd.testing.assert_frame_equal(pd.read_csv(saved_cohort),metadata.iloc[indices].reset_index(drop=True),check_dtype=False)
            cohort_hash=sha(saved_cohort)
            print(f'\n{split}: {len(cohort):,} rows, {cohort.group_id.nunique():,} groups, {cohort.ml_label.nunique()}/24 classes.',flush=True)

            for feature_set in pi.FEATURE_SETS:
                features=manifest['feature_sets'][feature_set];condition=f'{split}__{feature_set}'
                x=cohort[features].to_numpy(dtype=np.float64)
                for model in pi.MODELS:
                    for seed in pi.SEEDS:
                        key=f'{model}__{condition}__seed_{seed}';entry=entries[key]
                        folder=out/'checkpoints'/key;folder.mkdir(parents=True,exist_ok=True)
                        expected=['tree_path_dependent'] if model=='XGBoost' else BASELINES
                        results=[];missing=[]

                        for baseline in expected:
                            target=folder/f'{baseline}.json'
                            if target.is_file():
                                value=pi.read(target)
                                assert value['protocol_sha256']==lock_hash and value['model_sha256']==entry['model_sha256']
                                assert value['cohort_sha256']==cohort_hash and value['baseline']==baseline
                                results.append(value);continue
                            prior=None
                            if old_protocol_hash is not None:
                                prior_path=previous/'checkpoints'/key/f'{baseline}.json'
                                prior=valid_prior_result(prior_path,old_protocol_hash,entry,cohort_hash,condition,seed,baseline)
                            if prior is None:
                                missing.append(baseline);continue
                            source_folder=previous/'checkpoints'/key
                            source_npz=source_folder/('treeshap.npz' if baseline=='tree_path_dependent' else f'{baseline}.npz')
                            assert source_npz.is_file()
                            destination_npz=folder/source_npz.name;shutil.copy2(source_npz,destination_npz)
                            value=dict(prior);value['protocol_sha256']=lock_hash
                            value['imported_from_v1']=True;value['source_v1_protocol_sha256']=old_protocol_hash
                            value['attempts']=[dict(steps=value.get('steps'),seconds=value.get('seconds'),
                                p95_scaled_completeness_error=value.get('p95_scaled_completeness_error'))]
                            save(target,value);results.append(value);imported+=1
                            print(f'{key} / {baseline}: reused valid version-1 checkpoint.',flush=True)

                        if missing:
                            print(f'\n{key}: computing {missing}...',flush=True);verification_start=time.perf_counter()
                            assert pi.read(entry['labels'])==encoder.classes_.tolist()
                            pd.testing.assert_frame_equal(pd.read_csv(entry['metadata']),metadata,check_dtype=False)
                            with np.load(entry['predictions'],allow_pickle=False) as saved:
                                np.testing.assert_array_equal(saved['true_label'],yall);saved_prediction=saved['prediction'].copy()
                            probabilities,network,scaler,forward,tab=pilot.restore(torch,xgb,pi,entry,train,features,encoder,helpers)
                            cohort_probabilities=probabilities(x)
                            np.testing.assert_array_equal(cohort_probabilities.argmax(axis=1),saved_prediction[indices])
                            api_error=0.0
                            if model=='FT-Transformer':
                                api,_=helpers['ft1'].predict(torch,tab,scaler,x)
                                api_error=float(abs(api-cohort_probabilities).max())
                                np.testing.assert_allclose(api,cohort_probabilities,rtol=1e-5,atol=1e-6)
                            verification=dict(cohort_prediction_replay_passed=True,ft_api_probability_max_error=api_error,
                                              prediction_verification_seconds=time.perf_counter()-verification_start)

                            if model=='XGBoost':
                                assert missing==['tree_path_dependent']
                                attributes,diag=pilot.tree_attr(shap,xgb,network,x,y,features,folder)
                                value=dict(protocol_sha256=lock_hash,cohort_sha256=cohort_hash,model_sha256=entry['model_sha256'],
                                    model=model,condition=condition,split=split,feature_set=feature_set,seed=seed,
                                    feature_importance={f:float(v) for f,v in zip(features,abs(attributes).mean(axis=0))},
                                    imported_from_v1=False,attempts=[dict(steps=None,seconds=diag['seconds'])],**verification,**diag)
                                save(folder/'tree_path_dependent.json',value);results.append(value);computed+=1
                            else:
                                z=torch.from_numpy(scaler.transform(x).astype(np.float32))
                                references={'training_mean':np.asarray(scaler.mean_,dtype=np.float64),
                                    'training_median':np.median(train[features].to_numpy(dtype=np.float64),axis=0)}
                                for baseline in missing:
                                    raw_reference=references[baseline]
                                    bz=torch.from_numpy(scaler.transform(raw_reference[None,:]).astype(np.float32))
                                    attempts=[];previous_vector=None;accepted=None
                                    for steps in STEP_SCHEDULE:
                                        attributes,detail,diag=pilot.ig_attr(torch,IntegratedGradients,forward,z,bz,y,steps)
                                        vector=abs(attributes).mean(axis=0)
                                        relative_change=None if previous_vector is None else float(abs(vector-previous_vector).sum()/max(abs(vector).sum(),1e-12))
                                        attempt=dict(steps=steps,seconds=diag['seconds'],max_abs_completeness_error=diag['max_abs_completeness_error'],
                                            median_scaled_completeness_error=diag['median_scaled_completeness_error'],
                                            p95_scaled_completeness_error=diag['p95_scaled_completeness_error'],
                                            max_scaled_completeness_error=diag['max_scaled_completeness_error'],
                                            relative_L1_change_from_previous_steps=relative_change)
                                        attempts.append(attempt)
                                        print(f'  {baseline}, {steps} steps: p95 scaled error={diag["p95_scaled_completeness_error"]:.6g}',flush=True)
                                        if diag['p95_scaled_completeness_error']<=GATE:
                                            accepted=(attributes,detail,diag,vector,steps,relative_change);break
                                        previous_vector=vector
                                    if accepted is None:
                                        failure=dict(model=model,condition=condition,seed=seed,baseline=baseline,attempts=attempts)
                                        save(folder/f'{baseline}_numerical_failure.json',failure)
                                        raise RuntimeError(f'IG did not pass by 512 steps: {failure}')
                                    attributes,detail,diag,vector,steps,relative_change=accepted
                                    np.savez_compressed(folder/f'{baseline}.npz',attributions=attributes,raw_baseline=raw_reference,
                                        standardized_baseline=bz.numpy(),target=y,**detail)
                                    value=dict(protocol_sha256=lock_hash,cohort_sha256=cohort_hash,model_sha256=entry['model_sha256'],
                                        model=model,condition=condition,split=split,feature_set=feature_set,seed=seed,baseline=baseline,
                                        steps=steps,feature_importance={f:float(v) for f,v in zip(features,vector)},
                                        numerical_checks_passed=True,relative_L1_change_from_previous_steps=relative_change,
                                        imported_from_v1=False,attempts=attempts,**verification,**diag)
                                    save(folder/f'{baseline}.json',value);results.append(value);computed+=1
                            del probabilities,network,scaler,forward,tab,cohort_probabilities,saved_prediction
                            gc.collect()

                        assert len(results)==len(expected)
                        first=results[0]
                        replay.append(dict(model=model,condition=condition,seed=seed,rows=len(x),model_sha256=entry['model_sha256'],
                            cohort_prediction_replay_passed=first['cohort_prediction_replay_passed'],
                            ft_api_probability_max_error=first['ft_api_probability_max_error'],
                            seconds=first['prediction_verification_seconds']))
                        for result in results:
                            diagnostics.append(diagnostic_row(result))
                            for feature,value in result['feature_importance'].items():
                                rows.append(dict(model=model,condition=condition,split=split,feature_set=feature_set,seed=seed,
                                    baseline=result['baseline'],feature=feature,mean_abs_attribution=value))
                        pd.DataFrame(diagnostics).to_csv(out/'numerical_checks.csv',index=False)
                        pd.DataFrame(replay).to_csv(out/'cohort_prediction_replay.csv',index=False)

            del train,test,cohort,metadata
            gc.collect()

        v1.aggregate_and_compare(out,rows,pi_per_seed)
        assert len(rows)==450 and len(diagnostics)==60 and len(replay)==36
        imported_total=sum(bool(row.get('imported_from_v1',False)) for row in diagnostics)
        summary=dict(completed=True,training_performed=False,model_condition_seed_units=36,native_feature_rows=len(rows),
            numerical_units=len(diagnostics),checkpoints_imported_from_v1=imported_total,
            checkpoints_computed_in_v2=len(diagnostics)-imported_total,
            checkpoints_imported_during_this_process=imported,checkpoints_computed_during_this_process=computed,
            elapsed_seconds=time.perf_counter()-start,output_directory=str(out),
            next='Share summary.json, protocol.json, the five ranking/importance CSVs, numerical_checks.csv, and cohort_prediction_replay.csv.')
        save(out/'summary.json',summary)
        print('\nADAPTIVE FULL NATIVE ATTRIBUTION COMPLETE\n'+json.dumps(summary,indent=2),flush=True)
    except Exception:
        (out/'failure.txt').write_text(traceback.format_exc())
        print(f'Run stopped. Share failure.txt from {out}',flush=True)
        raise


if __name__=='__main__':
    main()
