"""Run the frozen JPCO targeted feature-ablation suite.

This runner performs 24 final fits:
  2 models x 4 frozen feature ablations x seeds 42, 43, 44.

It imports and hash-verifies the authoritative XGBoost and MLP stage-1
implementations.  Hyperparameters are inherited from the matching reference
feature set and are never selected using ablation validation/test results.
The completed XGBoost remove_start_time_ns seed-42 pilot is reused after its
protocol and artifact hashes are verified.  All other fits run sequentially.

The protocol is written before test evaluation.  The script is restart-safe:
completed workers are accepted only when their protocol and artifact hashes
match.  Results are descriptive paired ablations, not independent retuning.
"""

import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import random
import shutil
import sys
import time

for name in [
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
]:
    os.environ[name] = "2"

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight


EXPECTED = {
    "inputs": {
        "manifest.json": "f1eedfaa282d6ae90e1567d271f9baff06c62ba233aff5890ae5dad85cfe9605",
        "o3_primary.csv.gz": "9d7d7d217563ccb121eba466fc442d83d7f25495e6e5ba40ea9a0962114829c6",
        "split_summary.csv": "61de488a44179ca86205ffcb45e98baa9eb2d9a7b52a1ab13cf2845368f88d01",
        "tuning_random_split.csv.gz": "8afbc1e03efa547ae4d51dcd3982308ebaa67f018f2fc225b59daaa55f153342",
    },
    "scripts": {
        "xgboost_revision_stage1.py": "7c06aa5cffb280dea07ec88ce4c140e9d6a98db6574996b60a377d0946c78b7b",
        "mlp_revision_stage1.py": "1aed088a3fa04eaed98c5fa081328b606a3085b43c2b3bd71a44bf1f14d18b93",
    },
    "selections": {
        "xgb_original": (
            "xgboost_stage1_results/random_split__original_nine/selected_configuration.json",
            "aa3e88087fe36cec7c6a733b7c7e66e7758573d052708c0578f00ba3b4380660",
        ),
        "xgb_six": (
            "xgboost_stage1_results/random_split__timestamp_free_six/selected_configuration.json",
            "16018fc9e88becd36ac31eabbe913ab41c277445c0f319bd14d25bb6e5c1c60f",
        ),
        "mlp_original": (
            "mlp_stage1_results/random_split__original_nine/selected_configuration.json",
            "c2640a2888677b8f17b073db4ba6e50620f6b44a10c9177c98e232fd4a802e43",
        ),
        "mlp_six": (
            "mlp_stage1_results/random_split__timestamp_free_six/selected_configuration.json",
            "a77512a79daf8b3667c383e4d26590e7bccee6e3433ef1537a8a3611e0e0870a",
        ),
    },
    "baselines": {
        "XGBoost": (
            "xgboost_stage2_results/test_per_seed.csv",
            "f330921e6209deb24c31af2a3053e19df18cef53c5d4cb8f2333871feeb6ce72",
        ),
        "MLP": (
            "mlp_stage2_results/test_per_seed.csv",
            "ccc026e3d11c8bb9a699b698f434c853b345bc20d7f20b3be7233c8d3c5c5b15",
        ),
    },
}

ABLATIONS = {
    "remove_start_time_ns": {
        "reference": "original_nine",
        "features": ["peak_time", "peak_time_ns", "duration", "peak_frequency",
                     "central_freq", "amplitude", "snr", "q_value"],
        "removed": ["start_time_ns"],
    },
    "remove_peak_time_pair": {
        "reference": "original_nine",
        "features": ["start_time_ns", "duration", "peak_frequency", "central_freq",
                     "amplitude", "snr", "q_value"],
        "removed": ["peak_time", "peak_time_ns"],
    },
    "timestamp_free_remove_amplitude_snr": {
        "reference": "timestamp_free_six",
        "features": ["duration", "peak_frequency", "central_freq", "q_value"],
        "removed": ["amplitude", "snr"],
    },
    "timestamp_free_remove_q": {
        "reference": "timestamp_free_six",
        "features": ["duration", "peak_frequency", "central_freq", "amplitude", "snr"],
        "removed": ["q_value"],
    },
}

SEEDS = [42, 43, 44]
PILOT_PROTOCOL_SHA = "70677579fe36119dbc07f989945cc3f8c09273aad80254c4de198b6adb492486"
PILOT_ARTIFACTS = {
    "model.ubj": "896e71459a7993d4b7c5c4289be095217e2f8b39f1d06a5dc559d42663e26ad7",
    "label_mapping.json": "305af76438bb449d045b7530fffe619383948fdf607f0480942feddc35627d42",
    "validation_predictions.npz": "364fba8e7f049c0889fa4786b69e4e7e84936c5d3dac1fd0f25f3278140243b4",
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def load_module(name, path):
    module_spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def metric_dict(y_true, y_pred):
    return {
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_f1_all24": float(f1_score(
            y_true, y_pred, labels=np.arange(24), average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def write_class_metrics(path, y_true, y_pred, labels):
    report = classification_report(
        y_true, y_pred, labels=np.arange(24), target_names=labels,
        output_dict=True, zero_division=0,
    )
    rows = []
    for index, label in enumerate(labels):
        row = report[label]
        rows.append({
            "class_index": index,
            "class_label": label,
            "precision": row["precision"],
            "recall": row["recall"],
            "f1": row["f1-score"],
            "support": int(row["support"]),
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def locate_pilot(output_root):
    candidates = []
    for summary_path in output_root.rglob("pilot_summary.json"):
        try:
            summary = json.loads(summary_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        folder = summary_path.parent
        protocol_path = folder / "protocol.json"
        if (
            summary.get("model") == "XGBoost"
            and summary.get("condition") == "remove_start_time_ns"
            and summary.get("seed") == 42
            and protocol_path.is_file()
            and sha(protocol_path) == PILOT_PROTOCOL_SHA
        ):
            if all((folder / name).is_file() and sha(folder / name) == expected
                   for name, expected in PILOT_ARTIFACTS.items()):
                candidates.append(folder)
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one verified pilot folder under {output_root}; found {len(candidates)}: "
            + ", ".join(map(str, candidates))
        )
    return candidates[0]


def load_random_split(data_dir, features):
    columns = list(dict.fromkeys(
        ["source_record_id", "group_id", "ml_label", "random_split"] + features
    ))
    pieces = {"train": [], "validation": [], "test": []}
    for chunk in pd.read_csv(
        data_dir / "o3_primary.csv.gz", usecols=columns, chunksize=50000,
        float_precision="round_trip",
    ):
        for role in pieces:
            mask = chunk.random_split.eq(role)
            if mask.any():
                pieces[role].append(chunk.loc[mask].copy())
    frames = {role: pd.concat(parts, ignore_index=True) for role, parts in pieces.items()}
    counts = pd.read_csv(data_dir / "split_summary.csv")
    for role, frame in frames.items():
        expected_rows = int(counts.loc[
            (counts.configuration == "random_split") & (counts.partition == role), "rows"
        ].item())
        assert len(frame) == expected_rows and frame.source_record_id.is_unique
    assert set(frames["train"].group_id).isdisjoint(frames["validation"].group_id)
    assert set(frames["train"].group_id).isdisjoint(frames["test"].group_id)
    assert set(frames["validation"].group_id).isdisjoint(frames["test"].group_id)
    return frames


def fit_xgb(xgb, core, X, y, candidate, seed, features):
    rounds, params = core.parameters(candidate)
    params = dict(params)
    params["seed"] = seed
    weights = compute_sample_weight("balanced", y).astype(np.float32)
    started = time.perf_counter()
    matrix = xgb.DMatrix(
        X, label=y, weight=weights, feature_names=features, nthread=2,
    )
    model = xgb.train(params, matrix, num_boost_round=rounds, verbose_eval=False)
    return model, time.perf_counter() - started, rounds, params


def predict_xgb(xgb, model, X, features):
    matrix = xgb.DMatrix(X, feature_names=features, nthread=2)
    started = time.perf_counter()
    probabilities = model.predict(matrix)
    seconds = time.perf_counter() - started
    assert probabilities.shape == (len(X), 24)
    assert np.isfinite(probabilities).all()
    return probabilities.argmax(axis=1), seconds


def verify_completed_worker(folder, protocol_hash):
    result_path = folder / "result.json"
    if not result_path.is_file():
        return None
    result = json.loads(result_path.read_text())
    assert result["protocol_sha256"] == protocol_hash
    for name, expected_hash in result["artifact_sha256"].items():
        artifact = folder / name
        assert artifact.is_file() and sha(artifact) == expected_hash, f"Changed artifact: {artifact}"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    data_dir = root / "jpco_o3_splits_v1"
    output_root = root / "targeted_ablation_v1"
    suite_dir = output_root / "full_suite"
    suite_dir.mkdir(parents=True, exist_ok=True)

    print("Verifying frozen inputs, source implementations, selections and baselines...", flush=True)
    for relative, expected_hash in EXPECTED["inputs"].items():
        assert sha(data_dir / relative) == expected_hash, f"Input changed: {relative}"
    for relative, expected_hash in EXPECTED["scripts"].items():
        assert sha(root / relative) == expected_hash, f"Authoritative script changed: {relative}"
    selection_records = {}
    for key, (relative, expected_hash) in EXPECTED["selections"].items():
        path = root / relative
        assert sha(path) == expected_hash, f"Selection changed: {relative}"
        selection_records[key] = json.loads(path.read_text())
    baseline_frames = {}
    for model, (relative, expected_hash) in EXPECTED["baselines"].items():
        path = root / relative
        assert sha(path) == expected_hash, f"Baseline changed: {relative}"
        baseline_frames[model] = pd.read_csv(path)

    xgb_core = load_module("targeted_xgb_core", root / "xgboost_revision_stage1.py")
    mlp_core = load_module("targeted_mlp_core", root / "mlp_revision_stage1.py")
    import torch
    import xgboost as xgb
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)

    manifest = json.loads((data_dir / "manifest.json").read_text())
    assert manifest["feature_sets"]["original_nine"] == [
        "peak_time", "peak_time_ns", "start_time_ns", "duration",
        "peak_frequency", "central_freq", "amplitude", "snr", "q_value",
    ]
    assert manifest["feature_sets"]["timestamp_free_six"] == [
        "duration", "peak_frequency", "central_freq", "amplitude", "snr", "q_value",
    ]

    # Search the revision root because earlier pilot runners used more than one
    # output-parent convention. Exact protocol/artifact hashes make the match
    # unambiguous; location alone is never trusted.
    pilot_dir = locate_pilot(root)
    print(f"Verified reusable pilot: {pilot_dir}", flush=True)

    protocol = {
        "stage": "frozen targeted feature ablations with final grouped-random test evaluation",
        "version": "1.0",
        "script_sha256": sha(Path(__file__)),
        "inputs": EXPECTED["inputs"],
        "authoritative_scripts": EXPECTED["scripts"],
        "selection_artifacts": {
            key: {"path": relative, "sha256": digest}
            for key, (relative, digest) in EXPECTED["selections"].items()
        },
        "baseline_artifacts": {
            model: {"path": relative, "sha256": digest}
            for model, (relative, digest) in EXPECTED["baselines"].items()
        },
        "split": "frozen grouped-random 64/16/20 train/validation/test split",
        "conditions": ABLATIONS,
        "models": ["XGBoost", "MLP"],
        "seeds": SEEDS,
        "expected_fits": 24,
        "pilot_reuse": {
            "model": "XGBoost", "condition": "remove_start_time_ns", "seed": 42,
            "source_protocol_sha256": PILOT_PROTOCOL_SHA,
            "source_artifact_sha256": PILOT_ARTIFACTS,
        },
        "hyperparameter_rule": (
            "Original-nine configuration is inherited for the two timing ablations; "
            "timestamp-free-six configuration is inherited for amplitude/SNR and Q ablations. "
            "No ablation-specific retuning or validation-based selection."
        ),
        "xgboost": {
            "numeric_representation": "raw float32",
            "class_weighting": "balanced sample weights computed from full training partition",
            "reference_candidates": {"original_nine": 7, "timestamp_free_six": 8},
        },
        "mlp": {
            "numeric_representation": "training-fitted StandardScaler on float64, then float32 tensors",
            "epochs": 20, "batch_size": 1024, "activation": "Mish",
            "optimizer": "AdamW", "weight_decay": 0.01,
            "objective": "unweighted cross entropy",
            "reference_candidates": {"original_nine": 8, "timestamp_free_six": 11},
        },
        "test_policy": "One final evaluation of every frozen model/condition/seed; no selection from test results",
        "versions": {
            "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "sklearn": sklearn.__version__,
            "xgboost": xgb.__version__, "torch": torch.__version__,
        },
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "threads": 2,
    }
    protocol = json.loads(json.dumps(protocol))
    protocol_path = suite_dir / "protocol.json"
    if protocol_path.exists():
        assert json.loads(protocol_path.read_text()) == protocol, (
            "Existing full-suite protocol differs. Do not mix runs."
        )
    else:
        if any(suite_dir.iterdir()):
            raise RuntimeError("Full-suite folder contains files without a protocol.")
        save_json(protocol_path, protocol)
    protocol_hash = sha(protocol_path)
    print(f"Frozen protocol: {protocol_hash}", flush=True)

    all_features = manifest["feature_sets"]["original_nine"]
    print("Loading frozen grouped-random train, validation and test partitions...", flush=True)
    frames = load_random_split(data_dir, all_features)
    train, validation, test = frames["train"], frames["validation"], frames["test"]
    encoder = LabelEncoder().fit(train.ml_label)
    assert len(encoder.classes_) == 24
    assert set(validation.ml_label) <= set(encoder.classes_)
    assert set(test.ml_label) <= set(encoder.classes_)
    y_train = encoder.transform(train.ml_label)
    y_validation = encoder.transform(validation.ml_label)
    y_test = encoder.transform(test.ml_label)
    validation_ids = validation.source_record_id.astype(str).to_numpy(dtype=str)
    test_ids = test.source_record_id.astype(str).to_numpy(dtype=str)
    label_names = encoder.classes_.tolist()
    print(f"Train: {len(train):,}; validation: {len(validation):,}; test: {len(test):,}", flush=True)

    results = []
    total = 2 * len(ABLATIONS) * len(SEEDS)
    worker_index = 0
    for model_name in ["XGBoost", "MLP"]:
        model_slug = "xgboost" if model_name == "XGBoost" else "mlp"
        for condition, definition in ABLATIONS.items():
            features = definition["features"]
            reference = definition["reference"]
            candidate = {
                ("XGBoost", "original_nine"): 7,
                ("XGBoost", "timestamp_free_six"): 8,
                ("MLP", "original_nine"): 8,
                ("MLP", "timestamp_free_six"): 11,
            }[(model_name, reference)]
            X_train_dtype = np.float32 if model_name == "XGBoost" else np.float64
            X_train = train[features].to_numpy(dtype=X_train_dtype)
            X_validation = validation[features].to_numpy(dtype=X_train_dtype)
            X_test = test[features].to_numpy(dtype=X_train_dtype)
            assert np.isfinite(X_train).all() and np.isfinite(X_validation).all() and np.isfinite(X_test).all()

            for seed in SEEDS:
                worker_index += 1
                folder = suite_dir / "workers" / model_slug / condition / f"seed_{seed}"
                folder.mkdir(parents=True, exist_ok=True)
                print(f"\n[{worker_index}/{total}] {model_name}; {condition}; seed {seed}", flush=True)
                completed = verify_completed_worker(folder, protocol_hash)
                if completed is not None:
                    results.append(completed)
                    print("Verified completed worker; skipping retraining.", flush=True)
                    continue

                reused_pilot = (
                    model_name == "XGBoost"
                    and condition == "remove_start_time_ns"
                    and seed == 42
                )
                if model_name == "XGBoost":
                    model_path = folder / "model.ubj"
                    if reused_pilot:
                        shutil.copy2(pilot_dir / "model.ubj", model_path)
                        model = xgb.Booster()
                        model.load_model(model_path)
                        pilot_summary = json.loads((pilot_dir / "pilot_summary.json").read_text())
                        fit_seconds = float(pilot_summary["fit_seconds"])
                        validation_metrics = {
                            "weighted_f1": float(pilot_summary["validation_weighted_f1"]),
                            "macro_f1_all24": float(pilot_summary["validation_macro_f1_all24"]),
                            "accuracy": float(pilot_summary["validation_accuracy"]),
                        }
                        validation_pred, validation_seconds = predict_xgb(
                            xgb, model, X_validation, features)
                        replay = metric_dict(y_validation, validation_pred)
                        for key in validation_metrics:
                            assert abs(validation_metrics[key] - replay[key]) <= 1e-12
                        rounds, params = xgb_core.parameters(candidate)
                        params = dict(params); params["seed"] = seed
                    else:
                        model, fit_seconds, rounds, params = fit_xgb(
                            xgb, xgb_core, X_train, y_train, candidate, seed, features)
                        validation_pred, validation_seconds = predict_xgb(
                            xgb, model, X_validation, features)
                        validation_metrics = metric_dict(y_validation, validation_pred)
                        model.save_model(model_path)
                    test_pred, test_seconds = predict_xgb(xgb, model, X_test, features)
                    parameter_record = {"rounds": rounds, "params": params}
                    del model
                    artifact_names = ["model.ubj"]
                else:
                    prefix = f"  {condition}, seed {seed}"
                    model, scaler, history, fit_seconds = mlp_core.fit_model(
                        torch, X_train, y_train, candidate, seed, prefix)
                    validation_prob, validation_seconds = mlp_core.predict(
                        torch, model, scaler, X_validation)
                    test_prob, test_seconds = mlp_core.predict(torch, model, scaler, X_test)
                    validation_pred = validation_prob.argmax(axis=1)
                    test_pred = test_prob.argmax(axis=1)
                    validation_metrics = metric_dict(y_validation, validation_pred)
                    torch.save(model.state_dict(), folder / "state_dict.pt")
                    save_json(folder / "scaler.json", {
                        "features": features, "mean": scaler.mean_.tolist(),
                        "scale": scaler.scale_.tolist(), "var": scaler.var_.tolist(),
                        "n_samples_seen": int(scaler.n_samples_seen_),
                    })
                    pd.DataFrame(history).to_csv(folder / "training_history.csv", index=False)
                    parameter_record = dict(mlp_core.GRID[candidate])
                    parameter_record.update({"epochs": mlp_core.EPOCHS, "seed": seed})
                    artifact_names = ["state_dict.pt", "scaler.json", "training_history.csv"]
                    del model, scaler, history, validation_prob, test_prob

                test_metrics = metric_dict(y_test, test_pred)
                np.savez_compressed(
                    folder / "validation_predictions.npz",
                    source_record_id=validation_ids, y_true=y_validation, y_pred=validation_pred,
                )
                np.savez_compressed(
                    folder / "test_predictions.npz",
                    source_record_id=test_ids, y_true=y_test, y_pred=test_pred,
                )
                write_class_metrics(
                    folder / "test_class_metrics.csv", y_test, test_pred, label_names)
                save_json(folder / "configuration.json", {
                    "model": model_name, "condition": condition, "seed": seed,
                    "reference_feature_set": reference, "features": features,
                    "removed_features": definition["removed"], "candidate": candidate,
                    "parameters": parameter_record, "protocol_sha256": protocol_hash,
                    "reused_verified_pilot": reused_pilot,
                })
                artifact_names.extend([
                    "configuration.json", "validation_predictions.npz",
                    "test_predictions.npz", "test_class_metrics.csv",
                ])
                result = {
                    "model": model_name, "condition": condition, "seed": seed,
                    "reference_feature_set": reference, "feature_count": len(features),
                    "candidate": candidate, "reused_verified_pilot": reused_pilot,
                    "train_rows": len(y_train), "validation_rows": len(y_validation),
                    "test_rows": len(y_test), "fit_seconds": float(fit_seconds),
                    "validation_prediction_batch_seconds": float(validation_seconds),
                    "test_prediction_batch_seconds": float(test_seconds),
                    "validation_weighted_f1": validation_metrics["weighted_f1"],
                    "validation_macro_f1_all24": validation_metrics["macro_f1_all24"],
                    "validation_accuracy": validation_metrics["accuracy"],
                    "test_weighted_f1": test_metrics["weighted_f1"],
                    "test_macro_f1_all24": test_metrics["macro_f1_all24"],
                    "test_accuracy": test_metrics["accuracy"],
                    "protocol_sha256": protocol_hash,
                    "artifact_sha256": {name: sha(folder / name) for name in artifact_names},
                }
                save_json(folder / "result.json", result)
                results.append(result)
                print(json.dumps({
                    "validation_weighted_f1": result["validation_weighted_f1"],
                    "test_weighted_f1": result["test_weighted_f1"],
                    "test_macro_f1_all24": result["test_macro_f1_all24"],
                    "reused_verified_pilot": reused_pilot,
                }, indent=2), flush=True)
                del validation_pred, test_pred
                gc.collect()
            del X_train, X_validation, X_test
            gc.collect()

    assert len(results) == 24
    per_seed = pd.DataFrame(results).drop(columns=["artifact_sha256"])
    per_seed = per_seed.sort_values(["model", "condition", "seed"]).reset_index(drop=True)
    per_seed.to_csv(suite_dir / "test_per_seed.csv", index=False)

    metrics = ["test_weighted_f1", "test_macro_f1_all24", "test_accuracy"]
    summary_rows = []
    for (model, condition), group in per_seed.groupby(["model", "condition"], sort=True):
        row = {"model": model, "condition": condition, "seeds": len(group)}
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_sample_std"] = float(group[metric].std(ddof=1))
            row[f"{metric}_min"] = float(group[metric].min())
            row[f"{metric}_max"] = float(group[metric].max())
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(suite_dir / "test_summary.csv", index=False)

    difference_rows = []
    for row in per_seed.itertuples(index=False):
        reference_condition = f"random_split__{row.reference_feature_set}"
        baseline = baseline_frames[row.model]
        match = baseline.loc[
            (baseline.condition == reference_condition) & (baseline.seed == row.seed)
        ]
        assert len(match) == 1
        baseline_f1 = float(match.test_weighted_f1.iloc[0])
        difference_rows.append({
            "model": row.model, "condition": row.condition, "seed": row.seed,
            "reference_condition": reference_condition,
            "reference_test_weighted_f1": baseline_f1,
            "ablation_test_weighted_f1": row.test_weighted_f1,
            "ablation_minus_reference_weighted_f1": row.test_weighted_f1 - baseline_f1,
            "ablation_minus_reference_percentage_points": 100 * (row.test_weighted_f1 - baseline_f1),
        })
    differences = pd.DataFrame(difference_rows)
    differences.to_csv(suite_dir / "paired_differences.csv", index=False)
    difference_summary = differences.groupby(["model", "condition"], as_index=False).agg(
        mean_difference=("ablation_minus_reference_weighted_f1", "mean"),
        sample_std_difference=("ablation_minus_reference_weighted_f1", "std"),
        min_difference=("ablation_minus_reference_weighted_f1", "min"),
        max_difference=("ablation_minus_reference_weighted_f1", "max"),
    )
    difference_summary["mean_difference_percentage_points"] = 100 * difference_summary.mean_difference
    difference_summary.to_csv(suite_dir / "paired_difference_summary.csv", index=False)

    output_names = [
        "protocol.json", "test_per_seed.csv", "test_summary.csv",
        "paired_differences.csv", "paired_difference_summary.csv",
    ]
    final_summary = {
        "completed": True, "workers": 24, "models": ["XGBoost", "MLP"],
        "conditions": list(ABLATIONS), "seeds": SEEDS,
        "verified_pilot_reused": True, "test_evaluated": True,
        "protocol_sha256": protocol_hash,
        "output_sha256": {name: sha(suite_dir / name) for name in output_names},
        "output_directory": str(suite_dir),
        "next": (
            "Share summary.json, protocol.json, test_per_seed.csv, test_summary.csv, "
            "paired_differences.csv, and paired_difference_summary.csv."
        ),
    }
    save_json(suite_dir / "summary.json", final_summary)
    print("\nTARGETED ABLATION SUITE COMPLETE", flush=True)
    print(json.dumps(final_summary, indent=2), flush=True)
    print("\nPAIRED DIFFERENCE SUMMARY", flush=True)
    print(difference_summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
