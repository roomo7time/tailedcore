#!/bin/bash

for seed in {101..105}; do
    python make_tailed_noisy_mvtec.py \
        --data_name "mvtec" \
        --tail_type "step" \
        --noise_ratio 0.00 \
        --step_tail_k 4 \
        --step_tail_class_ratio 0.6 \
        --tail_level "random" \
        --seed ${seed}

    python make_tailed_noisy_mvtec.py \
        --data_name "mvtec" \
        --tail_type "step" \
        --noise_ratio 0.00 \
        --step_tail_k 1 \
        --step_tail_class_ratio 0.6 \
        --tail_level "random" \
        --seed ${seed}

    python make_tailed_noisy_mvtec.py \
        --data_name "mvtec" \
        --tail_type "pareto" \
        --noise_ratio 0.00 \
        --tail_level "random" \
        --seed ${seed}
done