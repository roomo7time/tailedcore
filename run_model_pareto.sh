#!/bin/bash

# Define config names
CONFIG_NAMES=(
    "atailedpatch_mvtec_01"
)

# Define specific data names with their corresponding seeds
DATA_NAMES=(
    "mvtec_pareto_nr10_seed0"
    "mvtec_pareto_nr10_seed2"
)

# Process each data name with each config name
for data_name in "${DATA_NAMES[@]}"; do
    for config_name in "${CONFIG_NAMES[@]}"; do
        python evaluate.py --data_sub_path "anomaly_detection/${data_name}" --config_name "${config_name}"
    done
done