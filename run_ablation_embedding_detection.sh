
#!/bin/bash

config_ids=(
    01
    02
    03
    04
    05
    06
    07
    08
    09
    10
    11
    12
    13
)

for config_id in "${config_ids[@]}"; do
    python ablation_embedding_detection.py \
        --data_sub_path "visa_step_random_nr05_tk4_tr60_seed101" \
        --config_name "ablationtailedpatch_mvtec_${config_id}"
done

