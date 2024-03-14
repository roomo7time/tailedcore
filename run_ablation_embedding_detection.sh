
#!/bin/bash

config_ids=(
    14
    15
    16
    17
    # 18
    # 19
    # 20
    # 21
    # 22
    # 23
    # 24
    # 25
    # 26
)

for config_id in "${config_ids[@]}"; do
    python ablation_embedding_detection.py \
        --data_sub_path "visa_step_random_nr05_tk4_tr60_seed101" \
        --config_name "ablationtailedpatch_mvtec_${config_id}"
done

