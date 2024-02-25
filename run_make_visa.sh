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