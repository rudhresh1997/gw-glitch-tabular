"""Validate the saved package without fitting a model. Run from the package folder."""
import json
from pathlib import Path
import pandas as pd

b=Path(__file__).resolve().parent
m=json.loads((b/'manifest.json').read_text())
p=pd.read_csv(b/'o3_primary.csv.gz',float_precision='round_trip')
a=pd.read_csv(b/'split_assignments.csv.gz')
d=pd.read_csv(b/'o3_master.csv.gz',usecols=['source_record_id','group_id','representative_record_id',
    'primary_eligible','conflicting_label_group','random_split','chronological_split','ifo','gravityspy_id','peak_gps_ns'])
assert len(d)==m['source_rows'] and len(p)==m['primary_rows']
assert d.source_record_id.is_unique and p.source_record_id.is_unique
assert d.primary_eligible.sum()==len(p)
assert set(d.loc[d.primary_eligible,'source_record_id'])==set(p.source_record_id)
assert set(d.representative_record_id)<=set(d.source_record_id)
assert d.conflicting_label_group.sum()==m['conflicting_records']
assert p[a.columns].reset_index(drop=True).equals(a)
for key in [['ifo','peak_gps_ns'],['gravityspy_id']]:
    assert d.groupby(key).group_id.nunique().max()==1
for config in ['random_split','chronological_split']:
    assert p.groupby('group_id')[config].nunique().max()==1
    ss={s:p[p[config].eq(s)] for s in ['train','validation','test']}
    for i,x in enumerate(ss):
        for y in list(ss)[i+1:]:
            assert not (set(ss[x].group_id)&set(ss[y].group_id))
            assert not pd.MultiIndex.from_frame(ss[x][m['feature_sets']['original_nine']]).isin(
                pd.MultiIndex.from_frame(ss[y][m['feature_sets']['original_nine']])).any()
    t=pd.read_csv(b/f'tuning_{config}.csv.gz')
    assert set(t.source_record_id)<=set(ss['train'].source_record_id)
    assert set(t.source_record_id)==set(p.loc[p.group_id.isin(t.group_id),'source_record_id'])
    assert len(t)==m['tuning_subsets'][config]['rows']
    if config=='chronological_split':
        assert ss['train'].peak_gps_ns.max()<ss['validation'].peak_gps_ns.min()
        assert ss['validation'].peak_gps_ns.max()<ss['test'].peak_gps_ns.min()
print('PASS: saved record counts, identities, group links, assignments, overlaps, chronology and tuning membership.')
