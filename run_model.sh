#!/bin/bash

# for seed in {444..499}; do
#     CONFIG_NAMES=(
#         "aatailedpatch_mvtec_01"
#     )

#     DATA_NAMES=(
#         "mvtec_step_hard_nr10_tk4_tr70_seed${seed}"
#     )

#     for data_name in "${DATA_NAMES[@]}"; do
#         for config_name in "${CONFIG_NAMES[@]}"; do
#             python evaluate.py --data_sub_path "${data_name}" --config_name "${config_name}"
#         done
#     done
# done


#!/bin/bash

DATA_NAME="mvtec_step_hard_nr10_tk4_tr70_seed"

for seed in {0..49}; do

    python evaluate.py \
        --data_sub_path "mvtec_pareto_random_nr10_seed${seed}" \
        --config_name "aatailedpatch_mvtec_02"

    python evaluate.py \
        --data_sub_path "mvtec_step_random_nr10_tk4_tr60_seed${seed}" \
        --config_name "aatailedpatch_mvtec_01"

    python evaluate.py \
        --data_sub_path "mvtec_step_random_nr10_tk1_tr60_seed${seed}" \
        --config_name "aatailedpatch_mvtec_01"
    
done