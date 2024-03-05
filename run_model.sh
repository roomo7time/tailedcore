
#!/bin/bash

seeds=(
    200
    203
    206
    212
    217
    218
    220
    221
    232
    237
    240
    241
    245
    247
)

# mvtec
# for seed in "${seeds[@]}"; do

#     # python evaluate.py \
#     #     --data_sub_path "mvtec_step_random_nr10_tk4_tr60_seed${seed}" \
#     #     --config_name "aatailedpatch_mvtec_09"

#     # python evaluate.py \
#     #     --data_sub_path "mvtec_step_random_nr10_tk1_tr60_seed${seed}" \
#     #     --config_name "aatailedpatch_mvtec_09"
    
#     python evaluate.py \
#         --data_sub_path "mvtec_pareto_random_nr10_seed${seed}" \
#         --config_name "aatailedpatch_mvtec_10"
    
# done

# visa
for seed in "${seeds[@]}"; do

    # python evaluate.py \
    #     --data_sub_path "visa_pareto_random_nr05_seed${seed}" \
    #     --config_name "aatailedpatch_mvtec_11"

    # python evaluate.py \
    #     --data_sub_path "visa_step_random_nr05_tk1_tr60_seed${seed}" \
    #     --config_name "aatailedpatch_mvtec_11"

    python evaluate.py \
        --data_sub_path "visa_step_random_nr05_tk4_tr60_seed${seed}" \
        --config_name "aatailedpatch_mvtec_11"
    
done