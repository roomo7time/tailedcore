#!/bin/bash





DATA_NAMES=("symlink_mvtec_pareto_nr10_seed0" "symlink_mvtec_step_nr10_k1_seed0" "symlink_mvtec_step_nr10_k4_seed7" "symlink_mvtec_pareto_nr10_seed2" "symlink_mvtec_step_nr10_k1_seed7")

for data_name in "${DATA_NAMES[@]}"; do
    python evaluate.py --data_sub_path "anomaly_detection/${data_name}"
done

