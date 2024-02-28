#!/bin/bash

for seed in {0..49}; do
    python make_tailed_noisy_mvtec.py \
        --tail_type "step" \
        --step_tail_k 4 \
        --step_tail_class_ratio 0.6 \
        --seed ${seed} \
        --tail_level "random"

    python make_tailed_noisy_mvtec.py \
        --tail_type "step" \
        --step_tail_k 1 \
        --step_tail_class_ratio 0.6 \
        --seed ${seed} \
        --tail_level "random"

    python make_tailed_noisy_mvtec.py \
        --tail_type "pareto" \
        --seed ${seed} \
        --tail_level "random"
done