"""Operational-complexity benchmark for the three frozen JPCO revision pipelines.

Scope: XGBoost, custom MLP and FT-Transformer; grouped-random split; original
nine features; seeds 42, 43 and 44. No training, tuning or model selection.

Each model is restored in a fresh subprocess. Saved full-test predictions are
replayed before timing. Timing starts with an in-memory raw numeric array and
includes model-specific preprocessing plus prediction, but excludes disk I/O
and model loading. Serialized sizes are exact file sizes. Process RSS includes
the Python/runtime stack and is not model-only memory. Neural parameter counts
and XGBoost tree/node/leaf counts are architecture-specific descriptors and
must not be interpreted as equivalent units of capacity.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

for _name in [
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
]:
    os.environ[_name] = "2"

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

MODELS = ["XGBoost", "MLP", "FT-Transformer"]
SEEDS = [42, 43, 44]
SPLIT = "random_split"
FEATURE_SET = "original_nine"
CONDITION = f"{SPLIT}__{FEATURE_SET}"
THREADS = 2
PI_SCRIPT_SHA = "031b03e548e65fdaa04be7686c05552ca1124d54903340c4517838e2439c2174"
PILOT_SCRIPT_SHA = "4e8417b1e89a8f052ab1e3b3984c8e414e427c036d34d8da834dd873bf692b3d"
PI_PROTOCOL_SHA = "e6f10e69d65df629c1a4de871417c228ac63e9404d3c3eeb5c667b0dbddd60ef"


def sha(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path | str):
    return json.loads(Path(path).read_text())


def save_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def load_module(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    assert specification.loader is not None
    specification.loader.exec_module(module)
    return module


def current_rss_mib() -> float:
    """Current resident set from macOS/Linux ps, in MiB."""
    value = subprocess.check_output(
        ["ps", "-o", "rss=", "-p", str(os.getpid())], text=True
    ).strip()
    return float(value) / 1024.0


def peak_rss_mib() -> float:
    """ru_maxrss is bytes on macOS and KiB on Linux."""
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / (1024.0**2 if sys.platform == "darwin" else 1024.0)


def package_versions() -> dict:
    import torch
    import xgboost as xgb

    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": version("scikit-learn"),
        "xgboost": xgb.__version__,
        "pytorch_tabular": version("pytorch-tabular"),
        "pytorch_lightning": version("pytorch-lightning"),
        "torchmetrics": version("torchmetrics"),
    }


def xgboost_structure(booster) -> dict:
    trees = [json.loads(item) for item in booster.get_dump(dump_format="json")]

    def visit(node: dict, depth: int = 0):
        if "leaf" in node:
            return 1, 1, 0, depth
        totals = [visit(child, depth + 1) for child in node.get("children", [])]
        nodes = 1 + sum(value[0] for value in totals)
        leaves = sum(value[1] for value in totals)
        splits = 1 + sum(value[2] for value in totals)
        maximum_depth = max([depth] + [value[3] for value in totals])
        return nodes, leaves, splits, maximum_depth

    counts = [visit(tree) for tree in trees]
    return {
        "parameter_count": None,
        "xgb_boosting_rounds": int(booster.num_boosted_rounds()),
        "xgb_tree_count": len(trees),
        "xgb_total_nodes": int(sum(item[0] for item in counts)),
        "xgb_total_leaves": int(sum(item[1] for item in counts)),
        "xgb_total_split_nodes": int(sum(item[2] for item in counts)),
        "xgb_max_tree_depth": int(max(item[3] for item in counts)),
    }


def neural_structure(network) -> dict:
    return {
        "parameter_count": int(sum(parameter.numel() for parameter in network.parameters())),
        "xgb_boosting_rounds": None,
        "xgb_tree_count": None,
        "xgb_total_nodes": None,
        "xgb_total_leaves": None,
        "xgb_total_split_nodes": None,
        "xgb_max_tree_depth": None,
    }


def timing_plan(cohort_rows: int) -> list[tuple[int, int, int]]:
    full = min(5000, cohort_rows)
    return [(1, 10, 100), (32, 8, 50), (256, 5, 30), (full, 3, 15)]


def measure_latency(probabilities, values: np.ndarray):
    rows = []
    for batch_size, warmups, repeats in timing_plan(len(values)):
        batch = np.ascontiguousarray(values[:batch_size])
        for _ in range(warmups):
            output = probabilities(batch)
            assert output.shape == (batch_size, 24) and np.isfinite(output).all()
        gc.collect()
        was_enabled = gc.isenabled()
        gc.disable()
        try:
            for repeat in range(repeats):
                start = time.perf_counter_ns()
                output = probabilities(batch)
                elapsed_ns = time.perf_counter_ns() - start
                assert output.shape == (batch_size, 24) and np.isfinite(output).all()
                seconds = elapsed_ns / 1e9
                rows.append(
                    {
                        "batch_size": batch_size,
                        "repeat": repeat,
                        "elapsed_seconds": seconds,
                        "microseconds_per_row": seconds * 1e6 / batch_size,
                        "rows_per_second": batch_size / seconds,
                    }
                )
        finally:
            if was_enabled:
                gc.enable()
    return rows


def summarize_timing(rows: list[dict]) -> list[dict]:
    frame = pd.DataFrame(rows)
    output = []
    for batch_size, group in frame.groupby("batch_size", sort=False):
        elapsed = group.elapsed_seconds.to_numpy()
        per_row = group.microseconds_per_row.to_numpy()
        throughput = group.rows_per_second.to_numpy()
        output.append(
            {
                "batch_size": int(batch_size),
                "timing_repeats": len(group),
                "latency_median_ms": float(np.median(elapsed) * 1000),
                "latency_p25_ms": float(np.quantile(elapsed, 0.25) * 1000),
                "latency_p75_ms": float(np.quantile(elapsed, 0.75) * 1000),
                "microseconds_per_row_median": float(np.median(per_row)),
                "throughput_rows_per_second_median": float(np.median(throughput)),
            }
        )
    return output


def bundle_paths(entry: dict) -> list[Path]:
    paths = [Path(entry["model_path"]), Path(entry["labels"]), Path(entry["selected_path"])]
    if entry["model"] != "XGBoost":
        paths.append(Path(entry["scaler"]))
    if entry["model"] == "FT-Transformer":
        paths.append(Path(entry["config"]))
    return paths


def worker(root: Path, model_name: str, seed: int) -> None:
    import torch
    import xgboost as xgb

    torch.set_num_threads(THREADS)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)

    out = root / "complexity_benchmark_v1"
    protocol_path = out / "protocol.json"
    protocol = read(protocol_path)
    protocol_hash = sha(protocol_path)
    assert protocol["script_sha256"] == sha(Path(__file__).resolve())
    assert protocol["versions"] == package_versions()

    pi_path = root / "common_permutation_importance.py"
    pilot_path = root / "native_attribution_pilot.py"
    assert sha(pi_path) == PI_SCRIPT_SHA
    assert sha(pilot_path) == PILOT_SCRIPT_SHA
    pi = load_module("complexity_pi", pi_path)
    pilot = load_module("complexity_pilot", pilot_path)
    helpers = pi.import_helpers(root)

    key = f"{model_name}__{CONDITION}__seed_{seed}"
    entry = protocol["jobs"][key]
    for path_text, expected in protocol["job_artifact_sha256"][key].items():
        assert sha(path_text) == expected, f"Changed artifact: {path_text}"

    base = root / "jpco_o3_splits_v1"
    manifest = read(base / "manifest.json")
    features = manifest["feature_sets"][FEATURE_SET]
    rss_process_start = current_rss_mib()
    train, test = pi.load_partitions(base, SPLIT, features)
    encoder = LabelEncoder().fit(train.ml_label)
    assert encoder.classes_.tolist() == read(entry["labels"])
    metadata = test[["source_record_id", "group_id", "peak_gps_ns", "ml_label"]].reset_index(drop=True)
    pd.testing.assert_frame_equal(pd.read_csv(entry["metadata"]), metadata, check_dtype=False)
    y_all = encoder.transform(test.ml_label)
    x_all = test[features].to_numpy(dtype=np.float64)
    cohort_indices = pi.cohort_indices(test, SPLIT)
    cohort = test.iloc[cohort_indices].copy()
    x_cohort = cohort[features].to_numpy(dtype=np.float64)
    saved_cohort = root / "permutation_importance_v1" / "cohorts" / f"{SPLIT}.csv.gz"
    assert sha(saved_cohort) == protocol["cohort_sha256"]
    pd.testing.assert_frame_equal(
        pd.read_csv(saved_cohort), metadata.iloc[cohort_indices].reset_index(drop=True), check_dtype=False
    )
    with np.load(entry["predictions"], allow_pickle=False) as stored:
        np.testing.assert_array_equal(stored["true_label"], y_all)
        saved_predictions = stored["prediction"].copy()
        if "source_record_id" in stored:
            np.testing.assert_array_equal(
                stored["source_record_id"], metadata.source_record_id.astype(str).to_numpy()
            )
    rss_after_data_load = current_rss_mib()

    restore_start = time.perf_counter()
    probabilities, network, scaler, _forward, tab = pilot.restore(
        torch, xgb, pi, entry, train, features, encoder, helpers
    )
    restore_seconds = time.perf_counter() - restore_start
    rss_after_model_restore = current_rss_mib()

    restored_all = probabilities(x_all).argmax(axis=1)
    np.testing.assert_array_equal(restored_all, saved_predictions)
    restored_cohort = probabilities(x_cohort).argmax(axis=1)
    np.testing.assert_array_equal(restored_cohort, saved_predictions[cohort_indices])

    structure = xgboost_structure(network) if model_name == "XGBoost" else neural_structure(network)
    paths = bundle_paths(entry)
    assert len(set(paths)) == len(paths) and all(path.is_file() for path in paths)
    model_bytes = Path(entry["model_path"]).stat().st_size
    bundle_bytes = sum(path.stat().st_size for path in paths)

    del train, test, metadata, cohort, x_all, y_all, restored_all, restored_cohort, saved_predictions
    del tab, scaler, _forward
    gc.collect()
    rss_loaded_model_after_data_release = current_rss_mib()

    timing_rows = measure_latency(probabilities, x_cohort)
    timing_summary = summarize_timing(timing_rows)
    rss_after_timing = current_rss_mib()

    test_metrics = entry["test_metrics"]
    result = {
        "protocol_sha256": protocol_hash,
        "model": model_name,
        "condition": CONDITION,
        "seed": seed,
        "cohort_rows": len(x_cohort),
        "cohort_sha256": protocol["cohort_sha256"],
        "model_sha256": entry["model_sha256"],
        "prediction_replay_passed": True,
        "test_weighted_f1": float(test_metrics["test_weighted_f1"]),
        "test_macro_f1_all24": float(test_metrics["test_macro_f1_all24"]),
        "model_file_bytes": model_bytes,
        "model_file_mib": model_bytes / 1024**2,
        "deployment_bundle_bytes": bundle_bytes,
        "deployment_bundle_mib": bundle_bytes / 1024**2,
        "deployment_bundle_files": [str(path) for path in paths],
        "restore_seconds_untimed": restore_seconds,
        "rss_process_start_mib": rss_process_start,
        "rss_after_data_load_mib": rss_after_data_load,
        "rss_after_model_restore_mib": rss_after_model_restore,
        "rss_loaded_model_after_data_release_mib": rss_loaded_model_after_data_release,
        "rss_after_timing_mib": rss_after_timing,
        "process_peak_rss_mib": peak_rss_mib(),
        "timing_scope": "raw in-memory numeric array through preprocessing and probability prediction; excludes disk I/O and model loading",
        "memory_scope": "fresh Python worker process; includes runtime/framework and allocator state; not model-only memory",
        **structure,
        "timing_summary": timing_summary,
        "timing_rows": timing_rows,
    }
    destination = out / "workers" / f"{model_name.replace('-', '').lower()}_seed_{seed}.json"
    save_json(destination, result)
    print(f"WORKER COMPLETE: {model_name}, seed {seed} -> {destination}", flush=True)


def aggregate(root: Path, protocol_hash: str) -> None:
    out = root / "complexity_benchmark_v1"
    per_seed_rows, timing_rows = [], []
    for model_name in MODELS:
        for seed in SEEDS:
            path = out / "workers" / f"{model_name.replace('-', '').lower()}_seed_{seed}.json"
            result = read(path)
            assert result["protocol_sha256"] == protocol_hash
            assert result["prediction_replay_passed"] is True
            timing = result.pop("timing_rows")
            summaries = result.pop("timing_summary")
            bundle_files = result.pop("deployment_bundle_files")
            result["deployment_bundle_file_count"] = len(bundle_files)
            per_seed_rows.append(result)
            for row in timing:
                timing_rows.append({"model": model_name, "seed": seed, **row})
            for summary in summaries:
                result_for_batch = {
                    "model": model_name,
                    "seed": seed,
                    **summary,
                }
                # Retained separately from raw repeats for a compact per-seed table.
                result_for_batch["condition"] = CONDITION
                timing_rows.append({"_summary_row": True, **result_for_batch})

    per_seed = pd.DataFrame(per_seed_rows)
    per_seed.to_csv(out / "complexity_per_seed.csv", index=False)
    raw_timing = pd.DataFrame([row for row in timing_rows if not row.get("_summary_row", False)])
    raw_timing.to_csv(out / "latency_repeats.csv", index=False)
    compact_timing = pd.DataFrame(
        [{k: v for k, v in row.items() if k != "_summary_row"} for row in timing_rows if row.get("_summary_row", False)]
    )
    compact_timing.to_csv(out / "latency_per_seed.csv", index=False)

    summary_rows = []
    metric_columns = [
        "test_weighted_f1", "model_file_mib", "deployment_bundle_mib",
        "rss_loaded_model_after_data_release_mib", "process_peak_rss_mib",
    ]
    for model_name, group in per_seed.groupby("model", sort=False):
        row = {"model": model_name, "condition": CONDITION, "training_seeds": len(group)}
        for metric in metric_columns:
            row[metric + "_mean"] = float(group[metric].mean())
            row[metric + "_sd"] = float(group[metric].std(ddof=1))
        for structural in [
            "parameter_count", "xgb_boosting_rounds", "xgb_tree_count",
            "xgb_total_nodes", "xgb_total_leaves", "xgb_total_split_nodes", "xgb_max_tree_depth",
        ]:
            values = group[structural].dropna()
            row[structural + "_mean"] = None if values.empty else float(values.mean())
            row[structural + "_sd"] = None if values.empty else float(values.std(ddof=1))
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(out / "complexity_summary.csv", index=False)

    latency_summary = (
        compact_timing.groupby(["model", "batch_size"], sort=False)
        .agg(
            training_seeds=("seed", "count"),
            latency_median_ms_mean=("latency_median_ms", "mean"),
            latency_median_ms_sd=("latency_median_ms", "std"),
            microseconds_per_row_mean=("microseconds_per_row_median", "mean"),
            microseconds_per_row_sd=("microseconds_per_row_median", "std"),
            throughput_rows_per_second_mean=("throughput_rows_per_second_median", "mean"),
            throughput_rows_per_second_sd=("throughput_rows_per_second_median", "std"),
        )
        .reset_index()
    )
    latency_summary.to_csv(out / "latency_summary.csv", index=False)

    summary = {
        "completed": True,
        "training_performed": False,
        "condition": CONDITION,
        "models": MODELS,
        "seeds": SEEDS,
        "workers": len(per_seed),
        "prediction_replays_passed": int(per_seed.prediction_replay_passed.sum()),
        "output_directory": str(out),
        "next": "Share summary.json, protocol.json, complexity_per_seed.csv, complexity_summary.csv, latency_per_seed.csv, and latency_summary.csv.",
    }
    save_json(out / "summary.json", summary)
    print("\nCOMPLEXITY BENCHMARK COMPLETE\n" + json.dumps(summary, indent=2), flush=True)


def parent(root: Path) -> None:
    import torch  # noqa: F401 - version and environment verification
    import xgboost as xgb  # noqa: F401

    root = root.expanduser().resolve()
    base = root / "jpco_o3_splits_v1"
    out = root / "complexity_benchmark_v1"
    worker_dir = out / "workers"
    worker_dir.mkdir(parents=True, exist_ok=True)

    pi_path = root / "common_permutation_importance.py"
    pilot_path = root / "native_attribution_pilot.py"
    assert sha(pi_path) == PI_SCRIPT_SHA, "Restore unchanged common_permutation_importance.py"
    assert sha(pilot_path) == PILOT_SCRIPT_SHA, "Restore unchanged native_attribution_pilot.py"
    assert sha(root / "permutation_importance_v1" / "protocol.json") == PI_PROTOCOL_SHA
    pi = load_module("complexity_parent_pi", pi_path)

    versions = package_versions()
    print("Locating and verifying all reviewed artifacts...", flush=True)
    found = pi.discover(root)
    entries, artifacts, inputs = pi.preflight(found, versions)
    for filename, expected in inputs.items():
        assert sha(base / filename) == expected, f"Changed input: {filename}"
    manifest = read(base / "manifest.json")
    cohort = root / "permutation_importance_v1" / "cohorts" / f"{SPLIT}.csv.gz"
    assert cohort.is_file()

    jobs, job_hashes = {}, {}
    for model_name in MODELS:
        for seed in SEEDS:
            key = f"{model_name}__{CONDITION}__seed_{seed}"
            entry = dict(entries[key])
            selected_path = found[model_name]["stage1"] / CONDITION / "selected_configuration.json"
            entry["selected_path"] = str(selected_path)
            jobs[key] = entry
            required = [
                Path(entry["model_path"]), Path(entry["predictions"]), Path(entry["metadata"]),
                Path(entry["labels"]), selected_path,
            ]
            if model_name != "XGBoost":
                required.append(Path(entry["scaler"]))
            if model_name == "FT-Transformer":
                required.append(Path(entry["config"]))
            job_hashes[key] = {str(path.resolve()): sha(path) for path in required}

    protocol = {
        "version": "1.0",
        "stage": "fresh-process operational complexity benchmark; no training or tuning",
        "script_sha256": sha(Path(__file__).resolve()),
        "source_script_sha256": {"common_permutation_importance.py": PI_SCRIPT_SHA, "native_attribution_pilot.py": PILOT_SCRIPT_SHA},
        "permutation_protocol_sha256": PI_PROTOCOL_SHA,
        "versions": versions,
        "platform": {"system": platform.system(), "release": platform.release(), "machine": platform.machine(), "processor": platform.processor()},
        "condition": CONDITION,
        "features": manifest["feature_sets"][FEATURE_SET],
        "models": MODELS,
        "seeds": SEEDS,
        "threads": THREADS,
        "device": "cpu",
        "cohort_source": "exact grouped-random 5000-group cohort frozen for common permutation importance",
        "cohort_sha256": sha(cohort),
        "timing": {
            "input": "in-memory raw numeric feature array",
            "includes": "model-specific scaling/tensor-or-DMatrix construction and probability prediction",
            "excludes": "disk I/O and model loading",
            "batch_plan": timing_plan(len(pd.read_csv(cohort))),
            "summary": "median and interquartile range within each fresh-process model/seed worker; mean and sample SD across three seeds",
        },
        "serialized_size": "exact bytes; model-only and reconstruction bundle",
        "memory": "current and peak fresh-process RSS; includes interpreter/framework/allocator and is not model-only",
        "structural_metrics": "neural parameter count and XGBoost rounds/trees/nodes/leaves/depth are family-specific and not equivalent",
        "scope_limitation": "three revised pipelines in the primary grouped-random nine-feature condition; not all ten submitted architectures",
        "training_performed": False,
        "verified_preflight_artifact_count": len(artifacts),
        "input_sha256": inputs,
        "jobs": jobs,
        "job_artifact_sha256": job_hashes,
    }
    lock = out / "protocol.json"
    if lock.exists():
        assert read(lock) == protocol, "Existing complexity protocol differs. Use a new output directory; do not mix runs."
    else:
        assert not any(out.iterdir()) or list(out.iterdir()) == [worker_dir], "Output directory contains untracked files."
        save_json(lock, protocol)
    protocol_hash = sha(lock)

    # Balanced order reduces systematic thermal/order bias across model families.
    order = [
        ("XGBoost", 42), ("MLP", 42), ("FT-Transformer", 42),
        ("MLP", 43), ("FT-Transformer", 43), ("XGBoost", 43),
        ("FT-Transformer", 44), ("XGBoost", 44), ("MLP", 44),
    ]
    for index, (model_name, seed) in enumerate(order, start=1):
        destination = worker_dir / f"{model_name.replace('-', '').lower()}_seed_{seed}.json"
        if destination.exists():
            existing = read(destination)
            assert existing["protocol_sha256"] == protocol_hash
            print(f"[{index}/9] verified completed worker: {model_name}, seed {seed}", flush=True)
            continue
        print(f"\n[{index}/9] fresh worker: {model_name}, seed {seed}", flush=True)
        command = [
            sys.executable, str(Path(__file__).resolve()), "--root", str(root),
            "--worker", "--model", model_name, "--seed", str(seed),
        ]
        subprocess.run(command, check=True, cwd=root)
        time.sleep(2)

    aggregate(root, protocol_hash)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    arguments = parser.parse_args()
    if arguments.worker:
        assert arguments.model is not None and arguments.seed is not None
        worker(arguments.root.expanduser().resolve(), arguments.model, arguments.seed)
    else:
        assert arguments.model is None and arguments.seed is None
        parent(arguments.root)


if __name__ == "__main__":
    main()
