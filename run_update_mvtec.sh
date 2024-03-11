#!/bin/bash

for seed in {101..105}; do
    # python update_tailed_noisy_mvtec.py \
    #     --old_data_name "mvtec_step_random_nr10_tk1_tr60_seed${seed}"\
    #     --data_name "mvtec" \
    #     --noise_ratio 0.1

    # python update_tailed_noisy_mvtec.py \
    #     --old_data_name "mvtec_step_random_nr10_tk4_tr60_seed${seed}"\
    #     --data_name "mvtec" \
    #     --noise_ratio 0.1 
    
    python update_tailed_noisy_mvtec.py \
        --old_data_name "mvtec_pareto_random_nr10_seed${seed}"\
        --data_name "mvtec" \
        --noise_ratio 0.4
done