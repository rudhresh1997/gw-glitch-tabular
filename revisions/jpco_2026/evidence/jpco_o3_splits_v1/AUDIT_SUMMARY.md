# O3 dataset and split audit — September 15, 2026

The dataset-and-split package is complete. It addresses the data-provenance and partitioning work for Referee 2, Comment 3, and prepares the timestamp ablations relevant to Referee 1. Model experiments and the response letter remain to be done.

## Population

- O3 source records preserved: **500,524**.
- Conflicting-label groups excluded from primary modeling: **18 groups / 39 records**.
- Additional nonconflicting duplicate copies collapsed: **62,732**.
- Primary benchmark: **437,753 rows in 437,750 groups**.
- All 24 catalog labels remain in the primary population.

These are source-provided ML labels. We have not adjudicated their scientific correctness. Seventeen conflicting groups contain matching numerical representations with different labels; one contains two feature vectors. The 39 records remain available for review.

## Frozen outer partitions

| Configuration | Training | Validation | Test |
|---|---:|---:|---:|
| Grouped random | 280,163 | 70,040 | 87,550 |
| Chronological | 280,161 | 70,041 | 87,551 |

The random partitions contain all 24 classes. Chronological training contains 24; validation and test each contain 23, with no `None_of_the_Above` examples. This is observed temporal coverage, not a reason to reshuffle future events.

Chronological boundaries use exact integer GPS nanoseconds: 1260331692937500000 and 1262848360812500000. Boundary timestamps belong to the later partition. All training timestamps precede validation, and all validation timestamps precede test.

## Checks

- Zero event-group overlap across partitions within each configuration.
- Zero identical original-nine-feature vectors crossing those partitions.
- Repeated Gravity Spy IDs and detector/timestamp groups stay together.
- Training-only tuning subsets: 32,000 rows for random; 32,002 rows for chronological, each covering 24 classes.
- Source hashes, versions, per-class counts, excluded records and reproducible build/check scripts are included.

Groups use exact detector/time, repeated Gravity Spy ID, and identical eleven-feature representations. This does not establish independence of nearby events or detector-state conditions. No temporal embargo was imposed. The same records can have different assignments across the two alternative experiments; this is intentional.

## How to use the package

Start with `o3_primary.csv.gz`, which contains the model-ready rows and both assignments. The larger `o3_master.csv.gz` retains all source records and image URLs. Load only the columns needed for the pilot on the 8 GB Mac. Do not train on identifiers, source metadata, confidence, group IDs or split names. Use the explicit feature lists in `manifest.json`.

This corrected population differs from the submitted row-wise benchmark. Report revised results as corrected experiments and explain the construction changes; do not silently substitute them into the old tables. Catalog values are authoritative for v1; exact numerical reconstruction of older processed CSVs remains incomplete.

**Next action:** prepare one CPU XGBoost pilot on the grouped-random training partition, with training-only tuning/validation and measured runtime and memory. Do not use the test partition for runtime-driven model selection. The full Mac workload is not yet benchmarked.
