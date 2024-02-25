#!/bin/bash


mkdir -p ./logs

for seed in {0..9}; do
    DATA_NAMES=(
        "visa_pareto_nr05_seed${seed}"
        "visa_step_nr05_tk1_tr60_seed${seed}"
        "visa_step_nr05_tk4_tr60_seed${seed}"
    )

    CONFIG_NAMES=(
        "tailedpatch_mvtec_01"
    )

    for data_name in "${DATA_NAMES[@]}"; do
        for config_name in "${CONFIG_NAMES[@]}"; do
            python extract_artifacts.py --data_sub_path "anomaly_detection/${data_name}" --config_name "${config_name}"
        done
    done
done
