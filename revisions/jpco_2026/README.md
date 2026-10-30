# JPCO 2026 Revision Reproducibility Package

This directory contains the curated code, notebooks, and lightweight
evidence used for the 2026 referee-driven revision of:

> Evaluating Machine-Learning Models for Multiclass Classification of
> LIGO Gravitational-Wave Glitches

The manuscript has been accepted for publication in *Journal of Physics
Communications* and assigned the following article DOI:

https://doi.org/10.1088/2399-6528/aeb8df

The final Version of Record may not yet be available.

The software archive uses the stable Zenodo concept DOI
[10.5281/zenodo.19475318](https://doi.org/10.5281/zenodo.19475318),
which resolves to the latest archived release. Version-specific Zenodo
DOIs identify immutable individual releases.

## Scope

The package documents the additional analyses performed during revision:

- duplicate-aware reconstruction of the Gravity Spy O3 cohort;
- frozen grouped-random and chronological evaluation protocols;
- temporal-generalization and timestamp-removal analyses;
- targeted feature ablations;
- native TreeSHAP and Integrated Gradients analyses;
- common permutation-importance comparisons;
- operational footprint and CPU-latency measurements;
- bandwidth augmentation;
- feature-provenance and historical-sampling audits;
- feasibility assessment for a matched image-based comparison; and
- generation of manuscript-facing numerical evidence tables.

## Directory layout

- `01_revision_experiments.ipynb`:
  temporal validation, attribution, permutation importance, and
  operational-complexity workflow.
- `02_targeted_feature_ablations.ipynb`:
  targeted XGBoost and MLP feature-ablation workflow.
- `03_feature_provenance_audit.ipynb`:
  source-catalogue, feature-transformation, and historical-sampling
  audits.
- `04_bandwidth_augmentation.ipynb`:
  timestamp-free six-feature versus bandwidth-augmented analyses.
- `05_image_comparison_audit.ipynb`:
  image-metadata overlap, split-alignment, and feasibility audits.
- `06_manuscript_evidence_tables.ipynb`:
  validation and formatting of manuscript-facing evidence tables.
- `*.py`:
  reviewed pilot, training, evaluation, attribution, and benchmarking
  scripts.
- `jpco_o3_splits_v1/`:
  split-construction and split-verification scripts.
- `evidence/`:
  lightweight result summaries, protocols, audit records, tables, and
  Figure 6 assets.
- `SCRIPT_SHA256SUMS.txt`:
  SHA-256 checksums for the copied revision scripts.
- `requirements-revision.txt`:
  pinned versions from the principal revision-analysis environment.

## Data and large-artifact policy

Raw Gravity Spy catalogues, reconstructed event-level datasets,
row-level split assignments, trained model checkpoints, serialized
models, prediction arrays, worker files, and execution logs are not
distributed here.

These exclusions avoid redistributing large data products and prevent
the repository from becoming a model-artifact archive. The included
manifests, hashes, protocols, summaries, and audit files document the
provenance and reported numerical results.

The frozen reconstructed cohort contains 437,753 O3 events after
duplicate-aware grouping and removal of 18 conflicting identity groups.
The source reconstruction began with 500,524 catalogue rows from the
four O3a/O3b H1/L1 source catalogues.

## Frozen evaluation policy

The principal revision comparison used:

- a grouped-random 64/16/20 train/validation/test split; and
- a chronological 64/16/20 train/validation/test split.

Identity groups were kept disjoint across partitions. Model selection
used training data only; the outer validation partition was not used to
select hyperparameters. Final test evaluation used seeds 42, 43, and 44.

The authoritative provenance records are:

- `evidence/jpco_o3_splits_v1/manifest.json`
- `evidence/jpco_o3_splits_v1/AUDIT_SUMMARY.md`
- `evidence/jpco_o3_splits_v1/split_summary.csv`

## Environment

The principal revision experiments were executed with Python 3.12.9 on
Apple Silicon using the package versions recorded in
`requirements-revision.txt`.

The split-construction manifest separately records the environment used
to construct the frozen cohort. Those construction versions are
provenance information; `requirements-revision.txt` describes the
principal model-analysis environment.

A notebook frontend was available outside the `env1` kernel environment,
so no particular Jupyter server implementation is pinned here.

Example environment setup:

```bash
conda create -n gw-glitch-revision python=3.12
conda activate gw-glitch-revision
python -m pip install -r requirements-revision.txt
```

A compatible Jupyter frontend or IDE must be installed separately if the
notebooks will be opened interactively.

## Portable path configuration

The notebook copies contain no machine-specific absolute home-directory paths.
Their bootstrap cells recognize these environment variables:

- `GW_GLITCH_REVISION_ROOT`: revision working directory containing the
  scripts, frozen split directory, and generated outputs.
- `GW_GLITCH_PROJECT_ROOT`: parent gravitational-wave glitch project.
- `GW_GLITCH_RESEARCH_ROOT`: parent research directory.
- `GW_GLITCH_DESKTOP_ROOT`: optional higher-level desktop directory used
  by bounded historical discovery cells.
- `GW_GLITCH_IMAGE_PROJECT_ROOT`: optional image-project directory used
  by the matched-image feasibility audit.

For workflows confined to the curated package, set the revision root to
this directory:

```bash
export GW_GLITCH_REVISION_ROOT="$(pwd)/revisions/jpco_2026"
```

The remaining variables are needed only for audit cells that inspect
external source catalogues, historical project files, or image metadata.
They should point to user-supplied local copies of those resources.

## Suggested execution order

The notebooks preserve the historical revision workflow rather than
forming a single push-button pipeline. A practical reading order is:

1. `01_revision_experiments.ipynb`
2. `02_targeted_feature_ablations.ipynb`
3. `03_feature_provenance_audit.ipynb`
4. `04_bandwidth_augmentation.ipynb`
5. `05_image_comparison_audit.ipynb`
6. `06_manuscript_evidence_tables.ipynb`

Many training cells require excluded event-level datasets and model
artifacts. The distributed evidence can be inspected without rerunning
those computational stages.

## Evidence integrity

Files copied verbatim from the frozen revision workspace are recorded in
`evidence/EVIDENCE_COPY_REPORT.json` and
`evidence/EVIDENCE_SHA256SUMS.txt`.

Eight path-bearing JSON records were converted into explicitly named
`*.public.json` derivatives. These derivatives preserve the scientific
content while replacing machine-specific path prefixes. The copy report
records both the original and public SHA-256 digests.

`PACKAGE_MANIFEST.json` provides a package-level inventory of distributed
files, byte sizes, and SHA-256 digests.

## Reproducibility boundary

The package supports inspection of the revision protocol, model
configurations, aggregate numerical results, attribution summaries,
ablation results, operational measurements, and provenance decisions.

Exact end-to-end retraining additionally requires the original Gravity
Spy source catalogues and reconstruction of the frozen event-level
cohort. Large trained models and prediction arrays are intentionally not
included. Therefore, this package should be described as a curated
revision reproducibility and audit record, not as a self-contained data
release.
