
#!/bin/bash

mkdir -p ./logs

for seed in {200..206}; do
    # Define config names
    CONFIG_NAMES=(
        "extract_mvtec_01"
        # "extract_mvtec_02"
        # "extract_mvtec_03"
        # "extract_mvtec_04"
    )

    # Define specific data names with their corresponding seeds
    DATA_NAMES=(
        "visa_step_random_nr05_tk1_tr60_seed${seed}"
        "visa_step_random_nr05_tk4_tr60_seed${seed}"
        "visa_pareto_random_nr05_seed${seed}"
    )

    # Process each data name with each config name
    for data_name in "${DATA_NAMES[@]}"; do
        for config_name in "${CONFIG_NAMES[@]}"; do
            python extract_artifacts.py \
                --data_sub_path "${data_name}" \
                --config_name "${config_name}"
        done
    done
done

