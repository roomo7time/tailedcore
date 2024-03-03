
#!/bin/bash

# mvtec
for seed in {101..105}; do

    python evaluate.py \
        --data_sub_path "mvtec_step_random_nr10_tk1_tr60_seed${seed}" \
        --config_name "aatailedpatch_mvtec_06"

    python evaluate.py \
        --data_sub_path "mvtec_step_random_nr10_tk4_tr60_seed${seed}" \
        --config_name "aatailedpatch_mvtec_06"
    
    python evaluate.py \
        --data_sub_path "mvtec_pareto_random_nr10_seed${seed}" \
        --config_name "aatailedpatch_mvtec_07"
    
done

# # visa
# for seed in {200..299}; do

#     python evaluate.py \
#         --data_sub_path "visa_pareto_random_nr05_seed${seed}" \
#         --config_name "aatailedpatch_mvtec_05"

#     python evaluate.py \
#         --data_sub_path "visa_step_random_nr05_tk4_tr60_seed${seed}" \
#         --config_name "aatailedpatch_mvtec_05"

#     python evaluate.py \
#         --data_sub_path "visa_step_random_nr05_tk1_tr60_seed${seed}" \
#         --config_name "aatailedpatch_mvtec_05"
    
# done