"""Read-only FT-Transformer environment/API check for the JPCO revision.

Run in a fresh env1 subprocess. Does not install packages, read datasets,
train models, load checkpoints, or change the completed XGBoost/MLP runs.
Writes one timestamped JSON report in the requested output directory.

Archive audit, September 16, 2026:
- PBS invokes fttransformer.py using PyTorch Tabular FTTransformerConfig.
- Optuna trial IDs 0..99: 32 completed, 68 pruned. Objective is 5-fold macro
  F1 on the whole sampled CSV, despite a printed 'weighted F1' label.
- Best completed trial 66: macro F1 0.455420634677372.
- CV allows 20 epochs, final allows 50, both early-stopping patience5;
  metric/checkpoint defaults require the installed framework's inspection.
- Final 64/16/20 split fixed at seed42. TrainerConfig gets seeds42..56 but
  fit(train=..., validation=...) omits seed. All framework seed messages say
  42, and all 15 printed scores are 0.7929. Do not treat zero displayed SD
  as evidence of independent-seed robustness. Inspect fit's actual behavior.
- Final F1 and prediction timing use a fixed unstratified sample of 10,112
  test events. The log says '100 samples' incorrectly. Metrics/attributions
  use the best test-scoring run, not the mean of the runs.
- Attribution refits a separate float32 scaler, uses a raw-zero baseline
  (unlike MLP zero in standardized space), and groups predicted-class logits
  by predicted label. Verify parity with the actual inference preprocessing
  and class mapping before reusing explanations. DeepLift is instantiated
  but only IG is evaluated. ROC/PR probability-column order is not explicitly
  matched to unique_classes in the script, so those curves need an audit.

Official fit API reference:
https://pytorch-tabular.readthedocs.io/en/latest/tabular_model/
"""
import argparse
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
import hashlib
import importlib
from importlib.metadata import version, PackageNotFoundError
import inspect
import json
import os
from pathlib import Path
import platform
import sys
import traceback

for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[name]='2'


def describe_config(cls):
    result={'signature':str(inspect.signature(cls))}
    if is_dataclass(cls):
        result['fields']=[{'name':f.name,'default':repr(f.default),
                           'default_factory':repr(f.default_factory)} for f in fields(cls)]
    return result


def describe_callable(obj):
    result={'signature':str(inspect.signature(obj))}
    try:
        result['source']=inspect.getsource(obj)
    except (OSError,TypeError):
        result['source']='Unavailable'
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    report={'purpose':'Read-only compatibility check; no training or data access',
            'python':platform.python_version(),'python_executable':sys.executable,
            'platform':platform.platform(),'architecture':platform.machine(),
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'packages':{},'test_evaluated':False,'training_started':False}
    packages=['torch','pytorch-tabular','pytorch-lightning','lightning','torchmetrics',
              'numpy','pandas','scikit-learn','omegaconf','einops','rich']
    for name in packages:
        try:
            report['packages'][name]=version(name)
        except PackageNotFoundError:
            report['packages'][name]=None
    print('Python:',sys.executable,flush=True)
    for name,value in report['packages'].items():
        print(f"{name}: {value if value is not None else 'not installed'}",flush=True)
    try:
        import torch
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        from pytorch_tabular import TabularModel
        from pytorch_tabular.models import FTTransformerConfig
        from pytorch_tabular.config import DataConfig,TrainerConfig,OptimizerConfig
        report['imports_ok']=True
        report['configs']={cls.__name__:describe_config(cls)
                           for cls in [DataConfig,TrainerConfig,OptimizerConfig,FTTransformerConfig]}
        report['tabular_model_constructor']=str(inspect.signature(TabularModel))
        report['methods']={}
        for name in ['fit','prepare_dataloader','prepare_model','train','predict','load_best_model']:
            obj=getattr(TabularModel,name,None)
            if obj is not None:
                report['methods'][name]=describe_callable(obj)
        # Inspect constructors only; no model is fitted or checkpoint loaded.
        report['ft_default_config_repr']=repr(FTTransformerConfig(task='classification'))
        report['trainer_default_config_repr']=repr(TrainerConfig())
        report['data_default_config_repr']=repr(DataConfig(target=['ml_label'],continuous_cols=['duration'],categorical_cols=[]))
        report['optimizer_default_config_repr']=repr(OptimizerConfig(optimizer='AdamW'))
        model_source=Path(inspect.getfile(TabularModel))
        report['tabular_model_source_sha256']=hashlib.sha256(model_source.read_bytes()).hexdigest()
        try:
            module=importlib.import_module('pytorch_tabular.models.ft_transformer.ft_transformer_model')
            report['ft_implementation']={}
            for name in ['FTTransformerBackbone','FTTransformerModel']:
                obj=getattr(module,name,None)
                if obj is not None:
                    report['ft_implementation'][name]=inspect.getsource(obj)
        except Exception:
            report['optional_ft_source_error']=traceback.format_exc()
        report['ready_for_pilot_preparation']=True
        print('\nImports and configuration inspection passed.',flush=True)
        print('TabularModel.fit:',report['methods']['fit']['signature'],flush=True)
    except Exception:
        report['ready_for_pilot_preparation']=False
        report.setdefault('imports_ok',False)
        report['error']=traceback.format_exc()
        print('\nCompatibility check encountered an error:\n'+report['error'],flush=True)
    out=args.output_dir.expanduser().resolve()
    out.mkdir(parents=True,exist_ok=True)
    name=datetime.now(timezone.utc).strftime('fttransformer_environment_%Y%m%dT%H%M%S_%fZ.json')
    path=out/name
    path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('\nREPORT SAVED:',path,flush=True)
    print('Share this JSON report. No package installation or training was performed.',flush=True)


if __name__=='__main__':
    main()
