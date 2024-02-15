#!/bin/bash



# DATA_NAMES=("RF10" "W330_W330A")

# for data_name in "${DATA_NAMES[@]}"; do
#     python evaluate.py --config_name "patchcore_${data_name}_08" --data_sub_path "hankook_tire/ml/${data_name}/version04" --data_format "labelme"
# done

DATA_NAMES=("LJ01" "RA33" "RF10" "W330_W330A")

for data_name in "${DATA_NAMES[@]}"; do
    python evaluate.py --config_name "patchcore_${data_name}_08" --data_sub_path "hankook_tire/ml/${data_name}/version03" --data_format "labelme"
done

# DATA_NAMES=("ALL")

# for data_name in "${DATA_NAMES[@]}"; do
#     python evaluate.py --config_name "patchcore_${data_name}_09" --data_sub_path "hankook_tire/ml/${data_name}/version02" --data_format "labelme"
# done


