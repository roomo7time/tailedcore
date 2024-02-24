#!/bin/bash

# Create logs directory if it doesn't exist
mkdir -p ./logs

python evaluate.py

for seed in {0..9}; do
    DATA_NAMES=(
        "symlink_mvtec_pareto_nr10_seed${seed}" 
        "symlink_mvtec_step_nr10_k1_seed${seed}" 
        "symlink_mvtec_step_nr10_k4_seed${seed}"
        "symlink_visa_pareto_nr10_seed${seed}"
        "symlink_visa_step_nr10_k1_seed${seed}"
        "symlink_visa_step_nr10_k4_seed${seed}"
    )

    CONFIG_NAMES=(
        "tailedpatch_mvtec_01"
        "tailedpatch_mvtec_05"
        "tailedpatch_mvtec_06"
        "tailedpatch_mvtec_07"
    )

    for data_name in "${DATA_NAMES[@]}"; do
        for config_name in "${CONFIG_NAMES[@]}"; do
            # python evaluate.py --data_sub_path "anomaly_detection/${data_name}" --config_name "${config_name}"
            python extract_artifacts.py --data_sub_path "anomaly_detection/${data_name}" --config_name "${config_name}"
            python analyze_extracted.py --data_sub_path "anomaly_detection/${data_name}" --config_name "${config_name}" # > "./logs/${data_name}_${config_name}_analyze_extracted.log" 2>&1
        done
    done
done
