"""Build O3 provenance, deduplicated benchmark and fixed outer splits.
Usage: python build_splits.py --input-dir /path/to/catalogs --output-dir ./rebuilt
Requires numpy, pandas and scikit-learn. No training or remote access.
"""
import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

FEATURES_11 = ['peak_time','peak_time_ns','start_time','start_time_ns','duration',
               'peak_frequency','central_freq','bandwidth','amplitude','snr','q_value']
FEATURES_9 = [c for c in FEATURES_11 if c not in ['start_time','bandwidth']]
FEATURES_6 = [c for c in FEATURES_9 if c not in ['peak_time','peak_time_ns','start_time_ns']]
SOURCES = ['H1_O3a.csv','H1_O3b.csv','L1_O3a.csv','L1_O3b.csv']


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-dir', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    args = ap.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    parts, hashes = [], {}
    for name in SOURCES:
        path = args.input_dir / name
        hashes[name] = sha256(path)
        cols = FEATURES_11 + ['ml_label','ifo','gravityspy_id','event_time','ml_confidence',
                             'channel','url1','url2','url3','url4']
        d = pd.read_csv(path, usecols=cols, float_precision='round_trip')
        d.insert(0, 'source_row', np.arange(len(d)))  # zero-based data row, excluding header
        d.insert(0, 'source_catalog', name)
        d.insert(0, 'source_record_id', [f'{Path(name).stem}:{i:06d}' for i in range(len(d))])
        parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    assert d[FEATURES_11+['ml_label','ifo','gravityspy_id']].notna().all().all()
    assert np.isfinite(d[FEATURES_11].to_numpy()).all()
    d['peak_gps_ns'] = d.peak_time.astype('int64')*1_000_000_000 + d.peak_time_ns.astype('int64')
    # Connected components: same detector+exact peak time OR same nonmissing Gravity Spy ID.
    # Also link identical full vectors across detectors, conservatively preventing that overlap.
    parent = np.arange(len(d))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(a,b):
        a,b = root(a),root(b)
        if a != b:
            parent[max(a,b)] = min(a,b)
    for keys in [['ifo','peak_gps_ns'], ['gravityspy_id'], FEATURES_11]:
        seen = {}
        for i,key in enumerate(d[keys].itertuples(index=False,name=None)):
            if key in seen:
                union(i,seen[key])
            else:
                seen[key] = i
    d['group_id'] = [f'g{root(i):07d}' for i in range(len(d))]
    nlabels = d.groupby('group_id').ml_label.nunique()
    conflict_ids = nlabels.index[nlabels.gt(1)]
    d['conflicting_label_group'] = d.group_id.isin(conflict_ids)
    # No majority-vote or confidence-based relabeling: preserve and quarantine whole groups.
    conflicts = d[d.conflicting_label_group].copy()
    conflicts.to_csv(out/'conflicting_records.csv',index=False)
    summary = conflicts.groupby('group_id').agg(
        records=('source_record_id','size'),labels=('ml_label',lambda x:';'.join(sorted(set(x)))),
        distinct_ids=('gravityspy_id','nunique'),ifo=('ifo','first'),
        peak_time=('peak_time','first'),peak_time_ns=('peak_time_ns','first'))
    summary.to_csv(out/'conflicting_groups.csv')
    # Collapse only exact detector + 11-feature + label copies, retaining all source links.
    dupkeys = ['ifo']+FEATURES_11+['ml_label']
    d['representative_record_id'] = d.groupby(dupkeys,sort=False,dropna=False).source_record_id.transform('first')
    d['primary_eligible'] = ~d.conflicting_label_group & d.source_record_id.eq(d.representative_record_id)
    p = d.loc[d.primary_eligible].copy()
    groups = p.groupby('group_id',sort=True).agg(
        ml_label=('ml_label','first'),rows=('source_record_id','size'),
        min_time=('peak_gps_ns','min'),max_time=('peak_gps_ns','max'))
    assert groups.groupby('ml_label').size().min() >= 3
    train,tmp = train_test_split(groups.index.to_numpy(),test_size=.36,
                                stratify=groups.ml_label,random_state=42)
    val,test = train_test_split(tmp,test_size=20/36,
                               stratify=groups.loc[tmp,'ml_label'],random_state=42)
    groups['random_split'] = ''
    for split,ids in [('train',train),('validation',val),('test',test)]:
        groups.loc[ids,'random_split'] = split
    # Choose chronological cutoffs at the 64% and 80% PRIMARY ROW positions.
    # Equal timestamps and any components straddling a cutoff go to the later partition.
    times = p.peak_gps_ns.sort_values().to_numpy()
    c1,c2 = int(times[int(.64*len(times))]),int(times[int(.8*len(times))])
    groups['chronological_split'] = np.select(
        [groups.max_time<c1,groups.max_time<c2],['train','validation'],default='test')
    # Components spanning cutoffs would violate strict temporal order; reject instead of hiding.
    for splitcol in ['random_split','chronological_split']:
        p[splitcol] = p.group_id.map(groups[splitcol])
        d[splitcol] = d.group_id.map(groups[splitcol]).fillna('excluded_conflicting_label')
    stats,counts,checks = [],[],{}
    for splitcol in ['random_split','chronological_split']:
        subsets = {s:p[p[splitcol].eq(s)] for s in ['train','validation','test']}
        for i,a in enumerate(subsets):
            for b in list(subsets)[i+1:]:
                overlap = set(subsets[a].group_id)&set(subsets[b].group_id)
                assert not overlap
                checks[f'{splitcol}_{a}_{b}_group_overlap'] = len(overlap)
                # Includes feature overlap after dropping start_time and bandwidth.
                x=pd.MultiIndex.from_frame(subsets[a][FEATURES_9])
                y=pd.MultiIndex.from_frame(subsets[b][FEATURES_9])
                n=int(x.isin(y).sum())
                assert n==0, ('identical features across splits',splitcol,a,b,n)
                checks[f'{splitcol}_{a}_{b}_identical_9feature_rows'] = n
        for s,t in subsets.items():
            stats.append(dict(configuration=splitcol,partition=s,rows=len(t),groups=t.group_id.nunique(),
                              fraction=len(t)/len(p),classes=t.ml_label.nunique(),
                              missing_classes=';'.join(sorted(set(p.ml_label)-set(t.ml_label))),
                              min_peak_gps_ns=int(t.peak_gps_ns.min()),max_peak_gps_ns=int(t.peak_gps_ns.max())))
            for label in sorted(p.ml_label.unique()):
                counts.append(dict(configuration=splitcol,partition=s,label=label,
                                   rows=int(t.ml_label.eq(label).sum())))
        if splitcol=='chronological_split':
            assert subsets['train'].peak_gps_ns.max()<subsets['validation'].peak_gps_ns.min()
            assert subsets['validation'].peak_gps_ns.max()<subsets['test'].peak_gps_ns.min()
            checks['strict_chronological_order'] = True
    # Membership table includes only model-ready primary representatives, not excluded rows.
    assignments=p[['source_record_id','group_id','random_split','chronological_split']].copy()
    assignments.to_csv(out/'split_assignments.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    # Training-only grouped tuning subset, approximately 32k primary rows (historical tuning scale).
    # Stratified groups from each outer TRAIN only. Temporal inner CV must still use time order.
    tuning_counts={}
    for config in ['random_split','chronological_split']:
        gt=groups[groups[config].eq('train')]
        k=min(len(gt)-1,max(2*gt.ml_label.nunique(),round(32000*len(gt)/gt.rows.sum())))
        chosen,_=train_test_split(gt.index.to_numpy(),train_size=k,stratify=gt.ml_label,random_state=20260915)
        mask=p.group_id.isin(chosen)
        assert p.loc[mask,config].eq('train').all()
        p.loc[mask,['source_record_id','group_id']].to_csv(out/f'tuning_{config}.csv.gz',index=False,
                                                       compression={'method':'gzip','mtime':0})
        tuning_counts[config]={'rows':int(mask.sum()),'groups':len(chosen),
                              'classes':int(p.loc[mask,'ml_label'].nunique())}
    # Don't export catalog class-probability columns: explicit predictor whitelists only.
    d.to_csv(out/'o3_master.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    primary_cols = ['source_record_id','source_catalog','source_row','group_id','ifo',
                    'gravityspy_id','peak_gps_ns']+FEATURES_11+['ml_label','random_split','chronological_split']
    p[primary_cols].to_csv(out/'o3_primary.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    pd.DataFrame(stats).to_csv(out/'split_summary.csv',index=False)
    pd.DataFrame(counts).to_csv(out/'class_counts.csv',index=False)
    manifest={'version':'1.0','source_sha256':hashes,'source_rows':len(d),'primary_rows':len(p),
              'source_groups':int(d.group_id.nunique()),'primary_groups':len(groups),
              'conflicting_groups':len(conflict_ids),'conflicting_records':len(conflicts),
              'duplicate_copies_collapsed_nonconflicting':int((~d.conflicting_label_group & ~d.primary_eligible).sum()),
              'feature_sets':{'original_nine':FEATURES_9,'timestamp_free_six':FEATURES_6,
                              'enriched_eleven':FEATURES_11},
              'target':'ml_label','chronological_cutoffs_gps_ns':[c1,c2],
              'random_seed':42,'tuning_seed':20260915,'tuning_subsets':tuning_counts,
              'checks':checks,'versions':{'python':platform.python_version(),'numpy':np.__version__,
                                        'pandas':pd.__version__,'scikit_learn':sklearn.__version__}}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))
    print(pd.DataFrame(stats).to_string(index=False))


if __name__=='__main__':
    main()
