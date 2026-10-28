# JPCO Revision Numerical Evidence

Generated exclusively from the hash-verified revision evidence freeze.

## Temporal generalization

| model | split | feature_set | train_rows | test_rows | supported_classes | seeds | weighted_f1_mean | weighted_f1_sd | macro_f1_all24_mean | macro_f1_all24_sd | accuracy_mean | accuracy_sd |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FT-Transformer | chronological_split | Original nine | 280161 | 87551 | 23 | 3 | 0.675730 | 0.020355 | 0.341126 | 0.016568 | 0.703796 | 0.018397 |
| FT-Transformer | chronological_split | Timestamp-free six | 280161 | 87551 | 23 | 3 | 0.688573 | 0.007527 | 0.360769 | 0.006374 | 0.721180 | 0.007210 |
| FT-Transformer | random_split | Original nine | 280163 | 87550 | 24 | 3 | 0.794598 | 0.005884 | 0.470303 | 0.005845 | 0.810032 | 0.001397 |
| FT-Transformer | random_split | Timestamp-free six | 280163 | 87550 | 24 | 3 | 0.780991 | 0.006198 | 0.446451 | 0.004869 | 0.801987 | 0.003107 |
| MLP | chronological_split | Original nine | 280161 | 87551 | 23 | 3 | 0.646178 | 0.011419 | 0.328817 | 0.019723 | 0.666747 | 0.015511 |
| MLP | chronological_split | Timestamp-free six | 280161 | 87551 | 23 | 3 | 0.715457 | 0.014453 | 0.417818 | 0.004098 | 0.736298 | 0.013951 |
| MLP | random_split | Original nine | 280163 | 87550 | 24 | 3 | 0.825389 | 0.003280 | 0.526317 | 0.009160 | 0.834285 | 0.002615 |
| MLP | random_split | Timestamp-free six | 280163 | 87550 | 24 | 3 | 0.814111 | 0.002969 | 0.486807 | 0.002246 | 0.826834 | 0.002414 |
| XGBoost | chronological_split | Original nine | 280161 | 87551 | 23 | 3 | 0.728064 | 0.001055 | 0.473613 | 0.000228 | 0.716497 | 0.002566 |
| XGBoost | chronological_split | Timestamp-free six | 280161 | 87551 | 23 | 3 | 0.699949 | 0.000823 | 0.455696 | 0.001862 | 0.650417 | 0.001137 |
| XGBoost | random_split | Original nine | 280163 | 87550 | 24 | 3 | 0.829742 | 0.000248 | 0.565175 | 0.002481 | 0.814152 | 0.000249 |
| XGBoost | random_split | Timestamp-free six | 280163 | 87550 | 24 | 3 | 0.825053 | 0.000311 | 0.558472 | 0.001664 | 0.809438 | 0.000200 |

## Temporal and timestamp effects

| analysis | model | feature_set | split | reference_weighted_f1 | comparison_weighted_f1 | difference | difference_percentage_points |
| --- | --- | --- | --- | --- | --- | --- | --- |
| timestamp_removal | XGBoost | Timestamp-free minus original | random_split | 0.829742 | 0.825053 | -0.004688 | -0.468847 |
| timestamp_removal | XGBoost | Timestamp-free minus original | chronological_split | 0.728064 | 0.699949 | -0.028115 | -2.811464 |
| timestamp_removal | MLP | Timestamp-free minus original | random_split | 0.825389 | 0.814111 | -0.011279 | -1.127882 |
| timestamp_removal | MLP | Timestamp-free minus original | chronological_split | 0.646178 | 0.715457 | 0.069280 | 6.927984 |
| timestamp_removal | FT-Transformer | Timestamp-free minus original | random_split | 0.794598 | 0.780991 | -0.013606 | -1.360626 |
| timestamp_removal | FT-Transformer | Timestamp-free minus original | chronological_split | 0.675730 | 0.688573 | 0.012843 | 1.284267 |
| chronological_minus_random | XGBoost | Original nine | Chronological minus random | 0.829742 | 0.728064 | -0.101678 | -10.167795 |
| chronological_minus_random | XGBoost | Timestamp-free six | Chronological minus random | 0.825053 | 0.699949 | -0.125104 | -12.510413 |
| chronological_minus_random | MLP | Original nine | Chronological minus random | 0.825389 | 0.646178 | -0.179212 | -17.921184 |
| chronological_minus_random | MLP | Timestamp-free six | Chronological minus random | 0.814111 | 0.715457 | -0.098653 | -9.865317 |
| chronological_minus_random | FT-Transformer | Original nine | Chronological minus random | 0.794598 | 0.675730 | -0.118868 | -11.886752 |
| chronological_minus_random | FT-Transformer | Timestamp-free six | Chronological minus random | 0.780991 | 0.688573 | -0.092419 | -9.241859 |

## Targeted feature ablations

| model | condition | reference_feature_set | seeds | reference_weighted_f1_mean | test_weighted_f1_mean | test_weighted_f1_sample_std | difference_percentage_points | test_macro_f1_all24_mean | test_accuracy_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP | remove_peak_time_pair | Original nine | 3 | 0.825389 | 0.815804 | 0.001886 | -0.958527 | 0.500850 | 0.826606 |
| MLP | remove_start_time_ns | Original nine | 3 | 0.825389 | 0.826858 | 0.004317 | 0.146863 | 0.523342 | 0.836101 |
| MLP | timestamp_free_remove_amplitude_snr | Timestamp-free six | 3 | 0.814111 | 0.783504 | 0.001464 | -3.060679 | 0.448419 | 0.798580 |
| MLP | timestamp_free_remove_q | Timestamp-free six | 3 | 0.814111 | 0.713189 | 0.006463 | -10.092134 | 0.422425 | 0.729572 |
| XGBoost | remove_peak_time_pair | Original nine | 3 | 0.829742 | 0.820622 | 0.000382 | -0.912012 | 0.550486 | 0.802597 |
| XGBoost | remove_start_time_ns | Original nine | 3 | 0.829742 | 0.829846 | 0.000203 | 0.010439 | 0.565985 | 0.814201 |
| XGBoost | timestamp_free_remove_amplitude_snr | Timestamp-free six | 3 | 0.825053 | 0.790680 | 0.000343 | -3.437348 | 0.509337 | 0.768909 |
| XGBoost | timestamp_free_remove_q | Timestamp-free six | 3 | 0.825053 | 0.773228 | 0.000525 | -5.182567 | 0.521774 | 0.754959 |

## Common permutation agreement

| condition | metric | model_pair | n_features | spearman_rho |
| --- | --- | --- | --- | --- |
| chronological_split__original_nine | macro_f1_all24 | MLP vs FT-Transformer | 9 | 0.966667 |
| chronological_split__original_nine | macro_f1_all24 | XGBoost vs FT-Transformer | 9 | 0.950000 |
| chronological_split__original_nine | macro_f1_all24 | XGBoost vs MLP | 9 | 0.983333 |
| chronological_split__original_nine | weighted_f1 | MLP vs FT-Transformer | 9 | 0.983333 |
| chronological_split__original_nine | weighted_f1 | XGBoost vs FT-Transformer | 9 | 0.966667 |
| chronological_split__original_nine | weighted_f1 | XGBoost vs MLP | 9 | 0.933333 |
| chronological_split__timestamp_free_six | macro_f1_all24 | MLP vs FT-Transformer | 6 | 0.885714 |
| chronological_split__timestamp_free_six | macro_f1_all24 | XGBoost vs FT-Transformer | 6 | 0.942857 |
| chronological_split__timestamp_free_six | macro_f1_all24 | XGBoost vs MLP | 6 | 0.942857 |
| chronological_split__timestamp_free_six | weighted_f1 | MLP vs FT-Transformer | 6 | 0.942857 |
| chronological_split__timestamp_free_six | weighted_f1 | XGBoost vs FT-Transformer | 6 | 1.000000 |
| chronological_split__timestamp_free_six | weighted_f1 | XGBoost vs MLP | 6 | 0.942857 |
| random_split__original_nine | macro_f1_all24 | MLP vs FT-Transformer | 9 | 0.983333 |
| random_split__original_nine | macro_f1_all24 | XGBoost vs FT-Transformer | 9 | 1.000000 |
| random_split__original_nine | macro_f1_all24 | XGBoost vs MLP | 9 | 0.983333 |
| random_split__original_nine | weighted_f1 | MLP vs FT-Transformer | 9 | 0.933333 |
| random_split__original_nine | weighted_f1 | XGBoost vs FT-Transformer | 9 | 0.950000 |
| random_split__original_nine | weighted_f1 | XGBoost vs MLP | 9 | 0.983333 |
| random_split__timestamp_free_six | macro_f1_all24 | MLP vs FT-Transformer | 6 | 0.885714 |
| random_split__timestamp_free_six | macro_f1_all24 | XGBoost vs FT-Transformer | 6 | 0.828571 |
| random_split__timestamp_free_six | macro_f1_all24 | XGBoost vs MLP | 6 | 0.828571 |
| random_split__timestamp_free_six | weighted_f1 | MLP vs FT-Transformer | 6 | 0.771429 |
| random_split__timestamp_free_six | weighted_f1 | XGBoost vs FT-Transformer | 6 | 0.771429 |
| random_split__timestamp_free_six | weighted_f1 | XGBoost vs MLP | 6 | 0.885714 |

## Permutation agreement ranges

| condition | metric | model_pairs | minimum_spearman_rho | maximum_spearman_rho | mean_spearman_rho |
| --- | --- | --- | --- | --- | --- |
| chronological_split__original_nine | macro_f1_all24 | 3 | 0.950000 | 0.983333 | 0.966667 |
| chronological_split__original_nine | weighted_f1 | 3 | 0.933333 | 0.983333 | 0.961111 |
| chronological_split__timestamp_free_six | macro_f1_all24 | 3 | 0.885714 | 0.942857 | 0.923810 |
| chronological_split__timestamp_free_six | weighted_f1 | 3 | 0.942857 | 1.000000 | 0.961905 |
| random_split__original_nine | macro_f1_all24 | 3 | 0.983333 | 1.000000 | 0.988889 |
| random_split__original_nine | weighted_f1 | 3 | 0.933333 | 0.983333 | 0.955556 |
| random_split__timestamp_free_six | macro_f1_all24 | 3 | 0.828571 | 0.885714 | 0.847619 |
| random_split__timestamp_free_six | weighted_f1 | 3 | 0.771429 | 0.885714 | 0.809524 |

## Operational complexity

| model | training_seeds | test_weighted_f1_mean | test_weighted_f1_sd | deployment_bundle_mib_mean | deployment_bundle_mib_sd | latency_median_ms_mean | latency_median_ms_sd | rss_loaded_model_after_data_release_mib_mean | rss_loaded_model_after_data_release_mib_sd | parameter_count_mean | xgb_tree_count_mean | xgb_total_leaves_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| XGBoost | 3 | 0.829742 | 0.000248 | 31.106592 | 0.096014 | 0.303702 | 0.008627 | 449.260417 | 22.122762 | -- | 4800.000000 | 435664.333333 |
| MLP | 3 | 0.825389 | 0.003280 | 0.663326 | 0.000007 | 0.100069 | 0.001066 | 461.567708 | 25.485833 | 172440.000000 | -- | -- |
| FT-Transformer | 3 | 0.794598 | 0.005884 | 0.511995 | 0.000028 | 0.759132 | 0.060857 | 529.729167 | 57.391819 | 127868.000000 | -- | -- |

## Bandwidth augmentation

| model | split | seeds | test_weighted_f1_mean | test_weighted_f1_sample_std | mean_difference | sample_std_difference | mean_difference_percentage_points | test_macro_f1_all24_mean | test_accuracy_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP | chronological_split | 3 | 0.712853 | 0.009236 | -0.002605 | 0.007183 | -0.260453 | 0.436947 | 0.736306 |
| MLP | random_split | 3 | 0.814582 | 0.009515 | 0.000471 | 0.011276 | 0.047101 | 0.493686 | 0.828715 |
| XGBoost | chronological_split | 3 | 0.706386 | 0.002065 | 0.006437 | 0.002600 | 0.643650 | 0.470438 | 0.660373 |
| XGBoost | random_split | 3 | 0.827284 | 0.000222 | 0.002231 | 0.000101 | 0.223111 | 0.559573 | 0.811715 |

