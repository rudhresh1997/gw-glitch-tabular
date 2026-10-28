"""Train one hash-locked XGBoost ablation pilot without reading the test set.

Condition: grouped-random split, remove only start_time_ns, seed 42.
The fitted model is retained for later frozen test evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder


EXPECTED_INPUT_SHA256 = {
    "manifest.json": "f1eedfaa282d6ae90e1567d271f9baff06c62ba233aff5890ae5dad85cfe9605",
    "o3_primary.csv.gz": "9d7d7d217563ccb121eba466fc442d83d7f25495e6e5ba40ea9a0962114829c6",
    "split_summary.csv": "61de488a44179ca86205ffcb45e98baa9eb2d9a7b52a1ab13cf2845368f88d01",
    "tuning_random_split.csv.gz": "8afbc1e03efa547ae4d51dcd3982308ebaa67f018f2fc225b59daaa55f153342",
}

EXPECTED_SELECTION_SHA256 = "aa3e88087fe36cec7c6a733b7c7e66e7758573d052708c0578f00ba3b4380660"

ORIGINAL_NINE = [
    "peak_time", "peak_time_ns", "start_time_ns", "duration",
    "peak_frequency", "central_freq", "amplitude", "snr", "q_value",
]

FEATURES = [
    "peak_time", "peak_time_ns", "duration", "peak_frequency",
    "central_freq", "amplitude", "snr", "q_value",
]

SEED = 42
TRAIN_ROWS = 280_163
VALIDATION_ROWS = 70_040


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text())


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True))


def package_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "xgboost": xgb.__version__,
    }


def load_train_validation(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = list(dict.fromkeys([
        "source_record_id", "group_id", "peak_gps_ns", "ml_label",
        "random_split", *FEATURES,
    ]))
    train_parts = []
    validation_parts = []
    for chunk in pd.read_csv(
        data_dir / "o3_primary.csv.gz",
        usecols=columns,
        float_precision="round_trip",
        chunksize=50_000,
    ):
        train_parts.append(chunk.loc[chunk.random_split.eq("train")].copy())
        validation_parts.append(chunk.loc[chunk.random_split.eq("validation")].copy())
    train = pd.concat(train_parts, ignore_index=True)
    validation = pd.concat(validation_parts, ignore_index=True)
    assert len(train) == TRAIN_ROWS
    assert len(validation) == VALIDATION_ROWS
    assert train.source_record_id.is_unique
    assert validation.source_record_id.is_unique
    assert set(train.source_record_id).isdisjoint(validation.source_record_id)
    assert set(train.group_id).isdisjoint(validation.group_id)
    assert set(train.ml_label) == set(validation.ml_label)
    return train, validation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    data_dir = root / "jpco_o3_splits_v1"
    selection_path = (
        root / "xgboost_stage1_results" /
        "random_split__original_nine" / "selected_configuration.json"
    )
    out = root / "targeted_ablation_v1" / "pilot_xgboost_remove_start_time_ns_seed42"
    out.mkdir(parents=True, exist_ok=True)

    print("Verifying frozen inputs and authoritative selection...", flush=True)
    observed_inputs = {}
    for name, expected in EXPECTED_INPUT_SHA256.items():
        path = data_dir / name
        assert path.exists(), f"Missing input: {path}"
        observed = sha256(path)
        assert observed == expected, f"Input changed: {name}"
        observed_inputs[name] = observed

    assert sha256(selection_path) == EXPECTED_SELECTION_SHA256, "Selection artifact changed."
    selected = read_json(selection_path)
    assert selected["candidate"] == 7
    assert selected["features"] == ORIGINAL_NINE
    if "selection_used_outer_validation" in selected:
        assert selected["selection_used_outer_validation"] is False
    if "test_evaluated" in selected:
        assert selected["test_evaluated"] is False

    params = dict(selected["params"])
    expected_params = {
        "objective": "multi:softprob", "num_class": 24,
        "tree_method": "hist", "device": "cpu", "nthread": 2,
        "seed": 42, "eval_metric": "mlogloss", "max_depth": 8,
        "eta": 0.05, "colsample_bytree": 0.8, "subsample": 0.8,
        "max_bin": 256, "min_child_weight": 1, "reg_lambda": 1.0,
        "reg_alpha": 0.0, "verbosity": 1,
    }
    assert params == expected_params
    rounds = int(selected.get("rounds", selected.get("num_boost_round", 200)))
    assert rounds == 200

    protocol = {
        "stage": "XGBoost targeted-ablation pilot; train and validation only",
        "condition": "remove_start_time_ns",
        "reference_feature_set": "original_nine",
        "split": "random_split",
        "features": FEATURES,
        "removed_features": ["start_time_ns"],
        "seed": SEED,
        "candidate": 7,
        "rounds": rounds,
        "params": params,
        "class_weight_rule": "training N / (24 * training class count)",
        "input_sha256": observed_inputs,
        "selection_path": str(selection_path),
        "selection_sha256": EXPECTED_SELECTION_SHA256,
        "script_sha256": sha256(Path(__file__).resolve()),
        "versions": package_versions(),
        "test_evaluated": False,
    }
    protocol_path = out / "protocol.json"
    if protocol_path.exists():
        assert read_json(protocol_path) == protocol, "Existing pilot protocol differs."
    else:
        unexpected = [path for path in out.iterdir() if path.name != "protocol.json"]
        assert not unexpected, f"Pilot output folder is not empty: {unexpected}"
        save_json(protocol_path, protocol)
    protocol_hash = sha256(protocol_path)

    summary_path = out / "pilot_summary.json"
    if summary_path.exists():
        summary = read_json(summary_path)
        assert summary["protocol_sha256"] == protocol_hash
        for name, expected in summary["artifact_sha256"].items():
            assert sha256(out / name) == expected, f"Changed pilot artifact: {name}"
        print("PILOT ALREADY COMPLETE AND VERIFIED", flush=True)
        print(json.dumps(summary, indent=2), flush=True)
        return

    print("Loading grouped-random training and validation partitions only...", flush=True)
    train, validation = load_train_validation(data_dir)
    encoder = LabelEncoder().fit(train.ml_label)
    assert len(encoder.classes_) == 24

    x_train = train[FEATURES].to_numpy(dtype=np.float32)
    x_validation = validation[FEATURES].to_numpy(dtype=np.float32)
    y_train = encoder.transform(train.ml_label)
    y_validation = encoder.transform(validation.ml_label)
    assert np.isfinite(x_train).all() and np.isfinite(x_validation).all()

    counts = np.bincount(y_train, minlength=24).astype(np.float64)
    assert np.all(counts > 0)
    class_weights = len(y_train) / (24.0 * counts)
    sample_weights = class_weights[y_train].astype(np.float32)

    dtrain = xgb.DMatrix(
        x_train, label=y_train, weight=sample_weights,
        feature_names=FEATURES,
    )
    dvalidation = xgb.DMatrix(
        x_validation, label=y_validation,
        feature_names=FEATURES,
    )

    print(
        f"Train: {len(train):,}; validation: {len(validation):,}; "
        f"features: {len(FEATURES)}; seed: {SEED}; rounds: {rounds}",
        flush=True,
    )
    print("Training starts; progress prints every 20 rounds.", flush=True)
    start = time.perf_counter()
    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=rounds,
        evals=[(dvalidation, "validation")],
        verbose_eval=20,
    )
    fit_seconds = time.perf_counter() - start

    probability = booster.predict(dvalidation)
    prediction = probability.argmax(axis=1)
    metrics = {
        "validation_weighted_f1": float(f1_score(
            y_validation, prediction, average="weighted", zero_division=0,
        )),
        "validation_macro_f1_all24": float(f1_score(
            y_validation, prediction, average="macro", labels=np.arange(24),
            zero_division=0,
        )),
        "validation_accuracy": float(accuracy_score(y_validation, prediction)),
    }

    model_path = out / "model.ubj"
    labels_path = out / "label_mapping.json"
    predictions_path = out / "validation_predictions.npz"
    booster.save_model(model_path)
    save_json(labels_path, encoder.classes_.tolist())
    np.savez_compressed(
        predictions_path,
        source_record_id=validation.source_record_id.astype(str).to_numpy(),
        true_label=y_validation,
        predicted_label=prediction,
        probabilities=probability,
    )

    summary = {
        "completed": True,
        "condition": "remove_start_time_ns",
        "model": "XGBoost",
        "seed": SEED,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "feature_count": len(FEATURES),
        "features": FEATURES,
        "fit_seconds": fit_seconds,
        **metrics,
        "model_mib": model_path.stat().st_size / (1024 ** 2),
        "protocol_sha256": protocol_hash,
        "artifact_sha256": {
            "model.ubj": sha256(model_path),
            "label_mapping.json": sha256(labels_path),
            "validation_predictions.npz": sha256(predictions_path),
        },
        "test_evaluated": False,
        "next": "Share protocol.json and pilot_summary.json; do not launch remaining fits yet.",
    }
    save_json(summary_path, summary)

    print("\nXGBOOST ABLATION PILOT COMPLETE", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
