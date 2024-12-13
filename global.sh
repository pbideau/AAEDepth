#!/bin/sh
# conda init
# conda activate v2e

# seq_list="0 1 2 3 4 5"
# seq_list="6 7 8 9"
seq_list="2"

# for seq in $seq_list; do
#     python3 ./src/main.py --dataset evimo --path /home/ncai/code/dataset/npz_samsung_mono_sfm/samsung_mono/sfm/eval/ --save ./src/output/adam --load_mask True --seq $seq --apply_mask True -f --dtime 0.05 --iter 10
# done


for seq in $seq_list; do
    python3 ./src/main.py --dataset evimo --path /Users/cainan/Desktop/active_perception/semester_poster/EVIMO/data/evimo_npz/samsung_mono/sfm/eval --save ./src/output/adam --load_mask True --seq $seq --apply_mask True -f --dtime 0.05 --iter 5
done