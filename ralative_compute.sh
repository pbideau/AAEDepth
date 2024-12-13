#!/bin/sh


# seq_list="0 1 2 3 4 5 6 7 8 9"
seq_list="4"

# for seq in $seq_list; do
#     python3 ./relative_distance/relative_dis_kalman.py \
#     --seq $seq \
#     --path ./dataset \
#     --estimate ./stppp_output
# done


for seq in $seq_list; do
    python3 ./relative_distance/relative_dis_kalman.py \
    --seq $seq \
    --path /Users/cainan/Desktop/active_perception/semester_poster/EVIMO/data/evimo_npz/samsung_mono/sfm/eval \
    --estimate /Users/cainan/Desktop/active_perception/event_depth_esti/code/stppp_and_output/output/evimo_marginalizeMagn_20_03_24/t_050_050_1
done