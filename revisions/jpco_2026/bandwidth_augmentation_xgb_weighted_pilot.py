
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb

from sklearn.metrics import (
    accuracy_score,
    f1_score,
)
from sklearn.utils.class_weight import (
    compute_sample_weight,
)


FEATURES = [
    "duration",
    "peak_frequency",
    "central_freq",
    "bandwidth",
    "amplitude",
    "snr",
    "q_value",
]

TARGET = "ml_label"
PARTITION_COLUMN = "random_split"

EXPECTED_INPUT_HASHES = {
    "manifest.json": (
        "f1eedfaa282d6ae90e1567d271f9baff"
        "06c62ba233aff5890ae5dad85cfe9605"
    ),
    "o3_primary.csv.gz": (
        "9d7d7d217563ccb121eba466fc442d83"
        "d7f25495e6e5ba40ea9a0962114829c6"
    ),
    "split_summary.csv": (
        "61de488a44179ca86205ffcb45e98baa"
        "9eb2d9a7b52a1ab13cf2845368f88d01"
    ),
    "selected_configuration.json": (
        "16018fc9e88becd36ac31eabbe913ab41"
        "c277445c0f319bd14d25bb6e5c1c60f"
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def save_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        required=True,
        type=Path,
    )

    args = parser.parse_args()
    root = args.root.expanduser().resolve()

    dataset_dir = root / "jpco_o3_splits_v1"

    manifest_path = (
        dataset_dir / "manifest.json"
    )
    primary_path = (
        dataset_dir / "o3_primary.csv.gz"
    )
    split_summary_path = (
        dataset_dir / "split_summary.csv"
    )

    selection_path = (
        root
        / "xgboost_stage1_results"
        / "random_split__timestamp_free_six"
        / "selected_configuration.json"
    )

    output_dir = (
        root
        / "bandwidth_augmentation_v1"
        / "pilot_weighted"
        / "random_split__timestamp_free_plus_bandwidth"
        / "xgboost_seed_42"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    input_paths = {
        "manifest.json": manifest_path,
        "o3_primary.csv.gz": primary_path,
        "split_summary.csv": split_summary_path,
        "selected_configuration.json": (
            selection_path
        ),
    }

    print("Verifying frozen inputs...")

    observed_hashes = {}

    for name, path in input_paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)

        observed = sha256_file(path)
        expected = EXPECTED_INPUT_HASHES[name]

        print(name)
        print("  observed:", observed)
        print("  expected:", expected)

        if observed != expected:
            raise RuntimeError(
                f"Input hash mismatch: {name}"
            )

        observed_hashes[name] = observed

    manifest = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )

    selected = json.loads(
        selection_path.read_text(
            encoding="utf-8"
        )
    )

    assert (
        selected["selection_used_outer_validation"]
        is False
    )

    assert selected["features"] == [
        "duration",
        "peak_frequency",
        "central_freq",
        "amplitude",
        "snr",
        "q_value",
    ]

    labels = selected["labels"]

    assert len(labels) == 24
    assert len(set(labels)) == 24

    label_to_index = {
        label: index
        for index, label in enumerate(labels)
    }

    required_columns = (
        FEATURES
        + [
            TARGET,
            PARTITION_COLUMN,
        ]
    )

    print()
    print(
        "Loading grouped-random training "
        "and validation partitions..."
    )

    frame = pd.read_csv(
        primary_path,
        usecols=required_columns,
        float_precision="round_trip",
    )

    assert not frame[
        FEATURES + [TARGET]
    ].isna().any().any()

    observed_partitions = set(
        frame[PARTITION_COLUMN].unique()
    )

    assert observed_partitions == {
        "train",
        "validation",
        "test",
    }

    train = frame.loc[
        frame[PARTITION_COLUMN] == "train"
    ].copy()

    validation = frame.loc[
        frame[PARTITION_COLUMN]
        == "validation"
    ].copy()

    # The test partition is deliberately
    # not materialized or evaluated.
    del frame

    assert len(train) == 280_163
    assert len(validation) == 70_040

    unknown_train_labels = (
        set(train[TARGET].unique())
        - set(labels)
    )
    unknown_validation_labels = (
        set(validation[TARGET].unique())
        - set(labels)
    )

    assert not unknown_train_labels
    assert not unknown_validation_labels

    X_train = train[FEATURES].to_numpy(
        dtype=np.float32,
        copy=True,
    )
    X_validation = validation[
        FEATURES
    ].to_numpy(
        dtype=np.float32,
        copy=True,
    )

    y_train = (
        train[TARGET]
        .map(label_to_index)
        .to_numpy(dtype=np.int32)
    )

    y_validation = (
        validation[TARGET]
        .map(label_to_index)
        .to_numpy(dtype=np.int32)
    )

    del train
    del validation

    parameters = dict(selected["params"])

    parameters.update(
        {
            "objective": "multi:softprob",
            "num_class": len(labels),
            "tree_method": "hist",
            "device": "cpu",
            "nthread": 2,
            "seed": 42,
            "eval_metric": "mlogloss",
            "verbosity": 1,
        }
    )

    rounds = int(selected["rounds"])

    protocol = {
        "experiment": (
            "timestamp_free_plus_bandwidth"
        ),
        "model": "XGBoost",
        "split": "random_split",
        "seed": 42,
        "training_performed": True,
        "test_evaluated": False,
        "source_configuration": str(
            selection_path
        ),
        "source_configuration_sha256": (
            observed_hashes[
                "selected_configuration.json"
            ]
        ),
        "source_feature_set": (
            "timestamp_free_six"
        ),
        "features": FEATURES,
        "feature_count": len(FEATURES),
        "target": TARGET,
        "partition_column": (
            PARTITION_COLUMN
        ),
        "train_rows": len(X_train),
        "validation_rows": len(
            X_validation
        ),
        "labels": labels,
        "parameters": parameters,
        "rounds": rounds,
        "class_weighting": (
            "balanced sample weights computed "
            "from the complete training partition"
        ),
        "input_sha256": observed_hashes,
        "manifest_version": manifest[
            "version"
        ],
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": (
                sklearn.__version__
            ),
            "xgboost": xgb.__version__,
        },
    }

    protocol_path = (
        output_dir / "protocol.json"
    )

    save_json(
        protocol_path,
        protocol,
    )

    protocol_sha256 = sha256_file(
        protocol_path
    )

    print()
    print(
        f"Train: {len(X_train):,}; "
        f"validation: "
        f"{len(X_validation):,}; "
        f"features: {len(FEATURES)}"
    )
    print(
        f"CPU threads: "
        f"{parameters['nthread']}; "
        f"rounds: {rounds}; "
        f"maximum depth: "
        f"{parameters['max_depth']}"
    )
    print(
        "Test partition will not be evaluated."
    )
    print(
        "Training starts; progress prints "
        "every 20 rounds."
    )

    training_weights = (
        compute_sample_weight(
            class_weight="balanced",
            y=y_train,
        )
        .astype(np.float32)
    )

    assert training_weights.shape == (
        len(y_train),
    )
    assert np.isfinite(
        training_weights
    ).all()
    assert (
        training_weights > 0
    ).all()

    dtrain = xgb.DMatrix(
        X_train,
        label=y_train,
        weight=training_weights,
        feature_names=FEATURES,
        nthread=2,
    )

    dvalidation = xgb.DMatrix(
        X_validation,
        label=y_validation,
        feature_names=FEATURES,
        nthread=2,
    )

    start = time.perf_counter()

    model = xgb.train(
        params=parameters,
        dtrain=dtrain,
        num_boost_round=rounds,
        evals=[
            (
                dvalidation,
                "validation",
            )
        ],
        verbose_eval=20,
    )

    fit_seconds = (
        time.perf_counter() - start
    )

    prediction_start = time.perf_counter()

    probabilities = model.predict(
        dvalidation
    )

    prediction_seconds = (
        time.perf_counter()
        - prediction_start
    )

    predictions = probabilities.argmax(
        axis=1
    ).astype(np.int32)

    weighted_f1 = f1_score(
        y_validation,
        predictions,
        average="weighted",
        zero_division=0,
    )

    macro_f1 = f1_score(
        y_validation,
        predictions,
        average="macro",
        labels=np.arange(len(labels)),
        zero_division=0,
    )

    accuracy = accuracy_score(
        y_validation,
        predictions,
    )

    model_path = output_dir / "model.ubj"
    model.save_model(model_path)

    label_mapping_path = (
        output_dir / "label_mapping.json"
    )

    save_json(
        label_mapping_path,
        {
            "labels": labels,
            "label_to_index": (
                label_to_index
            ),
        },
    )

    predictions_path = (
        output_dir
        / "validation_predictions.npz"
    )

    np.savez_compressed(
        predictions_path,
        y_true=y_validation,
        y_pred=predictions,
        probabilities=probabilities,
    )

    artifact_sha256 = {
        "model.ubj": sha256_file(
            model_path
        ),
        "label_mapping.json": sha256_file(
            label_mapping_path
        ),
        "validation_predictions.npz": (
            sha256_file(predictions_path)
        ),
    }

    summary = {
        "completed": True,
        "condition": (
            "random_split__"
            "timestamp_free_plus_bandwidth"
        ),
        "model": "XGBoost",
        "seed": 42,
        "train_rows": len(X_train),
        "validation_rows": len(
            X_validation
        ),
        "feature_count": len(FEATURES),
        "features": FEATURES,
        "fit_seconds": fit_seconds,
        "validation_predict_proba_"
        "batch_seconds": prediction_seconds,
        "validation_weighted_f1": (
            weighted_f1
        ),
        "validation_macro_f1_all24": (
            macro_f1
        ),
        "validation_accuracy": accuracy,
        "serialized_model_mib": (
            model_path.stat().st_size
            / (1024 ** 2)
        ),
        "protocol_sha256": (
            protocol_sha256
        ),
        "artifact_sha256": (
            artifact_sha256
        ),
        "test_evaluated": False,
        "output_directory": str(
            output_dir
        ),
        "next": (
            "Share protocol.json and "
            "pilot_summary.json. Do not "
            "launch the remaining 11 fits."
        ),
    }

    summary_path = (
        output_dir / "pilot_summary.json"
    )

    save_json(
        summary_path,
        summary,
    )

    print()
    print(
        "XGBOOST BANDWIDTH PILOT COMPLETE"
    )
    print(
        json.dumps(
            summary,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
