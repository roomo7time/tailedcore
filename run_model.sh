
#!/bin/bash

seeds=(
    101
    102
    103
    104
    105
)
# mvtec - ours: 9, 10 patchcore: 12
# visa - ours: 11 patchcore: 12

noise_ratios=(05 40 00)

for nr in "${noise_ratios[@]}"; do
    for seed in "${seeds[@]}"; do
        python evaluate.py \
            --data_sub_path "mvtec_step_random_nr${nr}_tk4_tr60_seed${seed}" \
            --config_name "aatailedpatch_mvtec_09"

        python evaluate.py \
            --data_sub_path "mvtec_step_random_nr${nr}_tk1_tr60_seed${seed}" \
            --config_name "aatailedpatch_mvtec_09"

        python evaluate.py \
            --data_sub_path "mvtec_pareto_random_nr${nr}_seed${seed}" \
            --config_name "aatailedpatch_mvtec_10"
    done
done

# # mvtec - 
# for seed in "${seeds[@]}"; do
#     python evaluate.py \
#         --data_sub_path "mvtec_step_random_nr05_tk4_tr60_seed${seed}" \
#         --config_name "aatailedpatch_mvtec_9"

#     python evaluate.py \
#         --data_sub_path "mvtec_step_random_nr05_tk1_tr60_seed${seed}" \
#         --config_name "aatailedpatch_mvtec_9"
    
#     python evaluate.py \
#         --data_sub_path "mvtec_pareto_random_nr05_seed${seed}" \
#         --config_name "aatailedpatch_mvtec_10"
# done

# # visa
# for seed in "${seeds[@]}"; do

#     # python evaluate.py \
#     #     --data_sub_path "visa_pareto_random_nr05_seed${seed}" \
#     #     --config_name "aatailedpatch_mvtec_11"

#     # python evaluate.py \
#     #     --data_sub_path "visa_step_random_nr05_tk1_tr60_seed${seed}" \
#     #     --config_name "aatailedpatch_mvtec_11"

#     python evaluate.py \
#         --data_sub_path "visa_step_random_nr05_tk4_tr60_seed${seed}" \
#         --config_name "aatailedpatch_mvtec_11"
    
# done