
#!/bin/bash

# Create logs directory if it doesn't exist
mkdir -p ./logs

# Define config names
CONFIG_NAMES=(
    "tailedpatch_mvtec_01"
    "tailedpatch_mvtec_05"
    "tailedpatch_mvtec_06"
    "tailedpatch_mvtec_07"
)

# Define specific data names with their corresponding seeds
DATA_NAMES=(
    "mvtec_step_nr10_tk1_tr60_seed0"
    "mvtec_step_nr10_tk1_tr60_seed7"
    "mvtec_step_nr10_tk4_tr60_seed0"
    "mvtec_step_nr10_tk4_tr60_seed7"
    "mvtec_pareto_nr10_seed0"
    "mvtec_pareto_nr10_seed2"
)

# Process each data name with each config name
for data_name in "${DATA_NAMES[@]}"; do
    for config_name in "${CONFIG_NAMES[@]}"; do
        python extract_artifacts.py --data_sub_path "anomaly_detection/${data_name}" --config_name "${config_name}"
        python analyze_extracted.py --data_sub_path "anomaly_detection/${data_name}" --config_name "${config_name}"
    done
done