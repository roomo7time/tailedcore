#!/bin/bash

# for seed in {0..50}; do
#     # python make_tailed_noisy_mvtec.py \
#     #     --source_dir "/home/jay/mnt/hdd01/data/image_datasets/anomaly_detection/mvtec" \
#     #     --tail_type "step" \
#     #     --step_tail_k 4 \
#     #     --seed ${seed}

#     python make_tailed_noisy_mvtec.py \
#         --data_name mvtec \
#         --tail_type "step" \
#         --step_tail_k 1 \
#         --source_dir "/home/jay/savespace/database/generic/mvtec" \
#         --seed ${seed}
#         --easy_tail False

#     # python make_tailed_noisy_mvtec.py \
#     #     --source_dir "/home/jay/mnt/hdd01/data/image_datasets/anomaly_detection/mvtec" \
#     #     --tail_type "pareto" \
#     #     --seed ${seed}
# done


python3 -m debugpy --listen 5678 --wait-for-client make_tailed_noisy_mvtec.py \
        --data_name mvtec \
        --tail_type "step" \
        --step_tail_k 1 \
        --source_dir "/home/jay/savespace/database/generic/mvtec" \
        --seed 0 \
        --easy_tail