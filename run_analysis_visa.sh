#!/bin/bash

for seed in {0..9}; do
    python make_tailed_noisy_mvtec.py \
        --source_dir "/home/jay/mnt/hdd01/data/image_datasets/anomaly_detection/visa" \
        --tail_type "step" \
        --step_tail_k 4 \
        --seed ${seed}

    python make_tailed_noisy_mvtec.py \
        --source_dir "/home/jay/mnt/hdd01/data/image_datasets/anomaly_detection/visa" \
        --tail_type "step" \
        --step_tail_k 1 \
        --seed ${seed}

    python make_tailed_noisy_mvtec.py \
        --source_dir "/home/jay/mnt/hdd01/data/image_datasets/anomaly_detection/visa" \
        --tail_type "pareto" \
        --seed ${seed}
done

mkdir -p ./logs

for seed in {0..9}; do
    DATA_NAMES=(
        "visa_pareto_nr10_seed${seed}"
        "visa_step_nr10_tk1_tr60_seed${seed}"
        "visa_step_nr10_tk4_tr60_seed${seed}"
    )

    CONFIG_NAMES=(
        "tailedpatch_mvtec_01"
    )

    for data_name in "${DATA_NAMES[@]}"; do
        for config_name in "${CONFIG_NAMES[@]}"; do
            python extract_artifacts.py --data_sub_path "anomaly_detection/${data_name}" --config_name "${config_name}"
            python analyze_extracted.py --data_sub_path "anomaly_detection/${data_name}" --config_name "${config_name}" # > "./logs/${data_name}_${config_name}_analyze_extracted.log" 2>&1
        done
    done
done
