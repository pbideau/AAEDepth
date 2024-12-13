#!/bin/sh

obj_list=$(seq 5000 1000 28000)

for obj in $obj_list; do
    echo "Processing object ID: $obj"
    python3 ./src/main.py --dataset evimo \
    --path ./dataset \
    --save ./stppp_output --load_mask True \
    --seq 4 --apply_mask True -f --dtime 0.05 --iter 100 --object_id $obj
done