#!/bin/bash

for seed in {30..49}; do
    # Define config names
    CONFIG_NAMES=(
        "aatailedpatch_mvtec_01"
    )

    # Define specific data names with their corresponding seeds
    DATA_NAMES=(
        "mvtec_step_nr10_tk1_tr70_seed${seed}"
        # "mvtec_step_nr10_tk4_tr70_seed${seed}"
        # "mvtec_pareto_nr10_seed${seed}"
    )

    # Process each data name with each config name
    for data_name in "${DATA_NAMES[@]}"; do
        for config_name in "${CONFIG_NAMES[@]}"; do
            python evaluate.py --data_sub_path "anomaly_detection/${data_name}" --config_name "${config_name}"
        done
    done
done
