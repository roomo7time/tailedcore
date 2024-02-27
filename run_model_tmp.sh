#!/bin/bash


# python evaluate.py --data_sub_path "mvtec_step_nr10_tk1_tr60_seed7" --config_name "tailedsoftpatch_mvtec_01"

python evaluate.py --data_sub_path "mvtec_step_hard_nr10_tk1_tr70_seed301" --config_name "aatailedpatch_mvtec_01"
python evaluate.py --data_sub_path "mvtec_step_hard_nr10_tk1_tr70_seed302" --config_name "aatailedpatch_mvtec_01"
python evaluate.py --data_sub_path "mvtec_step_hard_nr10_tk1_tr70_seed303" --config_name "aatailedpatch_mvtec_01"
python evaluate.py --data_sub_path "mvtec_step_hard_nr10_tk1_tr70_seed304" --config_name "aatailedpatch_mvtec_01"

