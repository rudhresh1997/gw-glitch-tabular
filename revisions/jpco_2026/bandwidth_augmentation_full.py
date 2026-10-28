"""Frozen bandwidth-augmentation experiment for the JPCO revision.

Runs 12 fresh final fits:
  XGBoost and MLP x random and chronological splits x seeds 42, 43, 44.

The only experimental change from the timestamp-free-six references is adding
the archived bandwidth column. Split-specific hyperparameters are inherited
from the completed training-only searches. No retuning or validation/test
selection is performed. The protocol is locked before any test prediction.
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


FEATURES = [
    "duration", "peak_frequency", "central_freq", "bandwidth",
    "amplitude", "snr", "q_value",
]
SPLITS = ["random_split", "chronological_split"]
SEEDS = [42, 43, 44]

EXPECTED = {
    "inputs": {
        "manifest.json": "f1eedfaa282d6ae90e1567d271f9baff06c62ba233aff5890ae5dad85cfe9605",
        "o3_primary.csv.gz": "9d7d7d217563ccb121eba466fc442d83d7f25495e6e5ba40ea9a0962114829c6",
        "split_summary.csv": "61de488a44179ca86205ffcb45e98baa9eb2d9a7b52a1ab13cf2845368f88d01",
    },
    "scripts": {
        "xgboost_revision_stage1.py": "7c06aa5cffb280dea07ec88ce4c140e9d6a98db6574996b60a377d0946c78b7b",
        "mlp_revision_stage1.py": "1aed088a3fa04eaed98c5fa081328b606a3085b43c2b3bd71a44bf1f14d18b93",
    },
    "selections": {
        ("XGBoost", "random_split"): (
            "xgboost_stage1_results/random_split__timestamp_free_six/selected_configuration.json",
            "16018fc9e88becd36ac31eabbe913ab41c277445c0f319bd14d25bb6e5c1c60f",
            8,
        ),
        ("XGBoost", "chronological_split"): (
            "xgboost_stage1_results/chronological_split__timestamp_free_six/selected_configuration.json",
            "8762ccde105a6019e455d9a70d82b9a460239438c233246c91f8a92a8cc4947e",
            2,
        ),
        ("MLP", "random_split"): (
            "mlp_stage1_results/random_split__timestamp_free_six/selected_configuration.json",
            "a77512a79daf8b3667c383e4d26590e7bccee6e3433ef1537a8a3611e0e0870a",
            11,
        ),
        ("MLP", "chronological_split"): (
            "mlp_stage1_results/chronological_split__timestamp_free_six/selected_configuration.json",
            "29730eb265893bb1216e5d0abee2711b74b315264b923387eea09306148b7174",
            8,
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


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metrics(y_true, y_pred):
    return {
        "weighted_f1": float(f1_score(
            y_true, y_pred, average="weighted", zero_division=0)),
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


def verify_worker(folder, protocol_hash):
    result_path = folder / "result.json"
    if not result_path.is_file():
        return None
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["protocol_sha256"] == protocol_hash
    for name, expected in result["artifact_sha256"].items():
        artifact = folder / name
        assert artifact.is_file() and sha(artifact) == expected, f"Changed artifact: {artifact}"
    return result


def load_primary(data_dir):
    columns = [
        "source_record_id", "group_id", "peak_gps_ns", "ml_label",
        "random_split", "chronological_split",
    ] + FEATURES
    parts = []
    for chunk in pd.read_csv(
        data_dir / "o3_primary.csv.gz", usecols=columns, chunksize=50000,
        float_precision="round_trip",
    ):
        parts.append(chunk)
    frame = pd.concat(parts, ignore_index=True)
    assert len(frame) == 437753
    assert frame.source_record_id.is_unique
    assert np.isfinite(frame[FEATURES].to_numpy(dtype=np.float64)).all()
    return frame


def split_frames(primary, split, split_summary):
    frames = {
        role: primary.loc[primary[split].eq(role)].reset_index(drop=True)
        for role in ["train", "validation", "test"]
    }
    for role, frame in frames.items():
        expected = int(split_summary.loc[
            (split_summary.configuration == split)
            & (split_summary.partition == role), "rows"
        ].item())
        assert len(frame) == expected and frame.source_record_id.is_unique
    assert set(frames["train"].group_id).isdisjoint(frames["validation"].group_id)
    assert set(frames["train"].group_id).isdisjoint(frames["test"].group_id)
    assert set(frames["validation"].group_id).isdisjoint(frames["test"].group_id)
    if split == "chronological_split":
        assert frames["train"].peak_gps_ns.max() < frames["validation"].peak_gps_ns.min()
        assert frames["validation"].peak_gps_ns.max() < frames["test"].peak_gps_ns.min()
    return frames


def predict_xgb(xgb, model, X):
    matrix = xgb.DMatrix(X, feature_names=FEATURES, nthread=2)
    started = time.perf_counter()
    probabilities = model.predict(matrix)
    elapsed = time.perf_counter() - started
    assert probabilities.shape == (len(X), 24)
    assert np.isfinite(probabilities).all()
    return probabilities.argmax(axis=1), elapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    data_dir = root / "jpco_o3_splits_v1"
    suite_dir = root / "bandwidth_augmentation_v1" / "full_suite"
    suite_dir.mkdir(parents=True, exist_ok=True)

    print("Verifying frozen inputs and implementations...", flush=True)
    for relative, expected in EXPECTED["inputs"].items():
        assert sha(data_dir / relative) == expected, f"Input changed: {relative}"
    for relative, expected in EXPECTED["scripts"].items():
        assert sha(root / relative) == expected, f"Script changed: {relative}"

    selections = {}
    selection_protocol = {}
    for key, (relative, expected, candidate) in EXPECTED["selections"].items():
        path = root / relative
        assert sha(path) == expected, f"Selection changed: {relative}"
        record = json.loads(path.read_text(encoding="utf-8"))
        assert record["candidate"] == candidate
        assert record["features"] == [
            "duration", "peak_frequency", "central_freq", "amplitude", "snr", "q_value"
        ]
        assert record["selection_used_outer_validation"] is False
        selections[key] = record
        selection_protocol[f"{key[0]}__{key[1]}"] = {
            "path": relative, "sha256": expected, "candidate": candidate,
        }

    baselines = {}
    baseline_protocol = {}
    for model, (relative, expected) in EXPECTED["baselines"].items():
        path = root / relative
        assert sha(path) == expected, f"Baseline changed: {relative}"
        baselines[model] = pd.read_csv(path)
        baseline_protocol[model] = {"path": relative, "sha256": expected}

    xgb_core = load_module("bandwidth_xgb_core", root / "xgboost_revision_stage1.py")
    mlp_core = load_module("bandwidth_mlp_core", root / "mlp_revision_stage1.py")
    import torch
    import xgboost as xgb
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)

    protocol = {
        "stage": "frozen timestamp-free bandwidth augmentation with final test evaluation",
        "version": "1.0",
        "script_sha256": sha(Path(__file__)),
        "inputs": EXPECTED["inputs"],
        "authoritative_scripts": EXPECTED["scripts"],
        "selection_artifacts": selection_protocol,
        "baseline_artifacts": baseline_protocol,
        "feature_condition": "timestamp_free_plus_bandwidth",
        "features": FEATURES,
        "reference_features": [
            "duration", "peak_frequency", "central_freq", "amplitude", "snr", "q_value"
        ],
        "added_feature": "bandwidth",
        "splits": {
            "random_split": "frozen grouped-random 64/16/20 split",
            "chronological_split": "frozen group-disjoint chronological 64/16/20 split",
        },
        "models": ["XGBoost", "MLP"],
        "seeds": SEEDS,
        "expected_fits": 12,
        "pilot_reuse": False,
        "hyperparameter_rule": (
            "Inherit the matching split-specific timestamp-free-six configuration; "
            "no bandwidth-specific retuning or validation/test selection."
        ),
        "xgboost": {
            "numeric_representation": "raw float32",
            "class_weighting": "balanced sample weights computed from full training partition",
        },
        "mlp": {
            "numeric_representation": "training-fitted StandardScaler on float64 then float32 tensors",
            "epochs": 20, "batch_size": 1024, "activation": "Mish",
            "optimizer": "AdamW", "weight_decay": 0.01,
            "objective": "unweighted cross entropy",
        },
        "test_policy": "One final evaluation for every frozen model/split/seed; no selection from test results",
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
        assert json.loads(protocol_path.read_text(encoding="utf-8")) == protocol, (
            "Existing protocol differs. Do not mix runs."
        )
    else:
        if any(suite_dir.iterdir()):
            raise RuntimeError("Full-suite directory contains files without a protocol.")
        save_json(protocol_path, protocol)
    protocol_hash = sha(protocol_path)
    print(f"Frozen protocol: {protocol_hash}", flush=True)

    split_summary = pd.read_csv(data_dir / "split_summary.csv")
    print("Loading frozen primary cohort...", flush=True)
    primary = load_primary(data_dir)
    results = []
    worker_index = 0

    for model_name in ["XGBoost", "MLP"]:
        model_slug = "xgboost" if model_name == "XGBoost" else "mlp"
        for split in SPLITS:
            frames = split_frames(primary, split, split_summary)
            train, validation, test = frames["train"], frames["validation"], frames["test"]
            encoder = LabelEncoder().fit(train.ml_label)
            assert len(encoder.classes_) == 24
            assert set(validation.ml_label) <= set(encoder.classes_)
            assert set(test.ml_label) <= set(encoder.classes_)
            labels = encoder.classes_.tolist()
            y_train = encoder.transform(train.ml_label)
            y_validation = encoder.transform(validation.ml_label)
            y_test = encoder.transform(test.ml_label)
            validation_ids = validation.source_record_id.astype(str).to_numpy(dtype=str)
            test_ids = test.source_record_id.astype(str).to_numpy(dtype=str)
            dtype = np.float32 if model_name == "XGBoost" else np.float64
            X_train = train[FEATURES].to_numpy(dtype=dtype)
            X_validation = validation[FEATURES].to_numpy(dtype=dtype)
            X_test = test[FEATURES].to_numpy(dtype=dtype)
            assert np.isfinite(X_train).all()
            assert np.isfinite(X_validation).all()
            assert np.isfinite(X_test).all()
            candidate = EXPECTED["selections"][(model_name, split)][2]

            for seed in SEEDS:
                worker_index += 1
                folder = suite_dir / "workers" / model_slug / split / f"seed_{seed}"
                folder.mkdir(parents=True, exist_ok=True)
                print(f"\n[{worker_index}/12] {model_name}; {split}; seed {seed}", flush=True)
                completed = verify_worker(folder, protocol_hash)
                if completed is not None:
                    results.append(completed)
                    print("Verified completed worker; skipping retraining.", flush=True)
                    continue

                random.seed(seed)
                np.random.seed(seed)
                if model_name == "XGBoost":
                    rounds, params = xgb_core.parameters(candidate)
                    params = dict(params)
                    params["seed"] = seed
                    weights = compute_sample_weight("balanced", y_train).astype(np.float32)
                    started = time.perf_counter()
                    matrix = xgb.DMatrix(
                        X_train, label=y_train, weight=weights,
                        feature_names=FEATURES, nthread=2,
                    )
                    model = xgb.train(
                        params, matrix, num_boost_round=rounds, verbose_eval=False,
                    )
                    fit_seconds = time.perf_counter() - started
                    validation_pred, validation_seconds = predict_xgb(
                        xgb, model, X_validation)
                    test_pred, test_seconds = predict_xgb(xgb, model, X_test)
                    model.save_model(folder / "model.ubj")
                    parameter_record = {"rounds": rounds, "params": params}
                    artifact_names = ["model.ubj"]
                    del model, matrix, weights
                else:
                    prefix = f"  {split}, seed {seed}"
                    model, scaler, history, fit_seconds = mlp_core.fit_model(
                        torch, X_train, y_train, candidate, seed, prefix)
                    validation_prob, validation_seconds = mlp_core.predict(
                        torch, model, scaler, X_validation)
                    test_prob, test_seconds = mlp_core.predict(
                        torch, model, scaler, X_test)
                    validation_pred = validation_prob.argmax(axis=1)
                    test_pred = test_prob.argmax(axis=1)
                    torch.save(model.state_dict(), folder / "state_dict.pt")
                    save_json(folder / "scaler.json", {
                        "features": FEATURES, "mean": scaler.mean_.tolist(),
                        "scale": scaler.scale_.tolist(), "var": scaler.var_.tolist(),
                        "n_samples_seen": int(scaler.n_samples_seen_),
                    })
                    pd.DataFrame(history).to_csv(folder / "training_history.csv", index=False)
                    parameter_record = dict(mlp_core.GRID[candidate])
                    parameter_record.update({"epochs": mlp_core.EPOCHS, "seed": seed})
                    artifact_names = ["state_dict.pt", "scaler.json", "training_history.csv"]
                    del model, scaler, history, validation_prob, test_prob

                validation_metrics = metrics(y_validation, validation_pred)
                test_metrics = metrics(y_test, test_pred)
                np.savez_compressed(
                    folder / "validation_predictions.npz",
                    source_record_id=validation_ids, y_true=y_validation, y_pred=validation_pred,
                )
                np.savez_compressed(
                    folder / "test_predictions.npz",
                    source_record_id=test_ids, y_true=y_test, y_pred=test_pred,
                )
                write_class_metrics(
                    folder / "test_class_metrics.csv", y_test, test_pred, labels)
                save_json(folder / "configuration.json", {
                    "model": model_name, "split": split, "seed": seed,
                    "reference_feature_set": "timestamp_free_six",
                    "feature_condition": "timestamp_free_plus_bandwidth",
                    "features": FEATURES, "added_feature": "bandwidth",
                    "candidate": candidate, "parameters": parameter_record,
                    "protocol_sha256": protocol_hash,
                })
                artifact_names.extend([
                    "configuration.json", "validation_predictions.npz",
                    "test_predictions.npz", "test_class_metrics.csv",
                ])
                result = {
                    "model": model_name, "split": split,
                    "condition": f"{split}__timestamp_free_plus_bandwidth",
                    "seed": seed, "feature_count": len(FEATURES),
                    "candidate": candidate, "train_rows": len(y_train),
                    "validation_rows": len(y_validation), "test_rows": len(y_test),
                    "fit_seconds": float(fit_seconds),
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
                }, indent=2), flush=True)
                del validation_pred, test_pred
                gc.collect()

            del X_train, X_validation, X_test
            gc.collect()

    assert len(results) == 12
    per_seed = pd.DataFrame(results).drop(columns=["artifact_sha256"])
    per_seed = per_seed.sort_values(["model", "split", "seed"]).reset_index(drop=True)
    per_seed.to_csv(suite_dir / "test_per_seed.csv", index=False)

    metric_names = ["test_weighted_f1", "test_macro_f1_all24", "test_accuracy"]
    summary_rows = []
    for (model, split), group in per_seed.groupby(["model", "split"], sort=True):
        row = {"model": model, "split": split, "seeds": len(group)}
        for metric in metric_names:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_sample_std"] = float(group[metric].std(ddof=1))
            row[f"{metric}_min"] = float(group[metric].min())
            row[f"{metric}_max"] = float(group[metric].max())
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(suite_dir / "test_summary.csv", index=False)

    difference_rows = []
    for row in per_seed.itertuples(index=False):
        reference_condition = f"{row.split}__timestamp_free_six"
        baseline = baselines[row.model]
        match = baseline.loc[
            (baseline.condition == reference_condition) & (baseline.seed == row.seed)
        ]
        assert len(match) == 1
        reference = float(match.test_weighted_f1.iloc[0])
        difference_rows.append({
            "model": row.model, "split": row.split, "seed": row.seed,
            "reference_condition": reference_condition,
            "augmented_condition": row.condition,
            "reference_test_weighted_f1": reference,
            "augmented_test_weighted_f1": row.test_weighted_f1,
            "augmented_minus_reference_weighted_f1": row.test_weighted_f1 - reference,
            "augmented_minus_reference_percentage_points": 100 * (row.test_weighted_f1 - reference),
        })
    differences = pd.DataFrame(difference_rows)
    differences.to_csv(suite_dir / "paired_differences.csv", index=False)
    difference_summary = differences.groupby(["model", "split"], as_index=False).agg(
        mean_difference=("augmented_minus_reference_weighted_f1", "mean"),
        sample_std_difference=("augmented_minus_reference_weighted_f1", "std"),
        min_difference=("augmented_minus_reference_weighted_f1", "min"),
        max_difference=("augmented_minus_reference_weighted_f1", "max"),
    )
    difference_summary["mean_difference_percentage_points"] = (
        100 * difference_summary.mean_difference
    )
    difference_summary.to_csv(suite_dir / "paired_difference_summary.csv", index=False)

    output_names = [
        "protocol.json", "test_per_seed.csv", "test_summary.csv",
        "paired_differences.csv", "paired_difference_summary.csv",
    ]
    final = {
        "completed": True, "workers": 12,
        "models": ["XGBoost", "MLP"], "splits": SPLITS, "seeds": SEEDS,
        "feature_condition": "timestamp_free_plus_bandwidth",
        "pilot_reused": False, "test_evaluated": True,
        "protocol_sha256": protocol_hash,
        "output_sha256": {name: sha(suite_dir / name) for name in output_names},
        "output_directory": str(suite_dir),
        "next": (
            "Share summary.json, protocol.json, test_per_seed.csv, test_summary.csv, "
            "paired_differences.csv, and paired_difference_summary.csv."
        ),
    }
    save_json(suite_dir / "summary.json", final)
    print("\nBANDWIDTH AUGMENTATION SUITE COMPLETE", flush=True)
    print(json.dumps(final, indent=2), flush=True)
    print("\nPAIRED DIFFERENCE SUMMARY", flush=True)
    print(difference_summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
