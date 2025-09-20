# Active Event Alignment for Monocular Distance Estimation

This is the implementation of the paper "Active Event Alignment for Monocular Distance Estimation". The code takes events as input. 

link to the paper: [https://openaccess.thecvf.com/content/WACV2025/html/Cai_Active_Event_Alignment_for_Monocular_Distance_Estimation_WACV_2025_paper.html](https://openaccess.thecvf.com/content/WACV2025/html/Cai_Active_Event_Alignment_for_Monocular_Distance_Estimation_WACV_2025_paper.html)

Authors: Nan Cai, Pia Bideau

<p align="center">
  <img height="220" src="/imgs/wacv-overview.png">
</p>


<p align="center">
  <img height="450" src="/imgs/result_EVIMO2sfm.gif">
</p>


## Setup

### High-level Input-Output


**Input**:
- Events.
- Semantic segmentation mask
- IMU angular velocity

**Output**:
- Relative distance of different objects (region)

## Usage

### Requirements
    pandas==1.0.5
    matplotlib==3.2.1
    numpy==1.18.5
    opencv_python==4.2.0.34
    torch==1.8.0
    scikit_learn==0.24.2
    scipy==1.4.1

### Download datasets
- [EVIMO2 dataset](https://better-flow.github.io/evimo/download_evimo_2.html): In our paper we use [samsung_mono sequences](https://obj.umiacs.umd.edu/evimo2v2npz/npz_samsung_mono_sfm.tar.gz) of the Structure from Motion type.
  - we include an example EVIMO2 data for the test


### Test
First, run the ST-PPP algorithm to estimate the "virtual" angular velocity of the event camera for every object in the segmentation mask:

<!-- say global and object alignment -->

    python3 ./src/main.py --dataset evimo \
    --path ./dataset \
    --save ./stppp_output \
    --load_mask True \
    --seq 4 --apply_mask True -f --dtime 0.05 --iter 100


Second, loop through all the object IDs using the following script:

    sh obj.sh




Then, use the ST-PPP results to estimate the relative distance of each object:

    python3 ./relative_distance/relative_dis_kalman.py \
    --seq 4 \
    --path ./dataset \
    --estimate ./stppp_output


 The relative depth figure would be saved to ```save_dir/current_date/sequence```


## Result Structure 

### rel_res_frame_dict.npy
This file contains a dictionary where each key is a timestamp (formatted as a string) and each value is another dictionary. The inner dictionary maps object IDs to their relative distances at that specific timestamp.

Example structure:
```python
{
    "0.000000": {
        5000: 1.0,
        6000: 0.8,
        ...
    },
    "0.050000": {
        5000: 1.1,
        6000: 0.85,
        ...
    },
    ...
}
```

### rel_res_obj_dict.npy
This file contains a dictionary where each key is an object ID and each value is another dictionary. The inner dictionary maps timestamps (formatted as strings) to the relative distances of the object at those specific timestamps.

Example structure:
```python
{
    5000: {
        "0.000000": 1.0,
        "0.050000": 1.1,
        ...
    },
    6000: {
        "0.000000": 0.8,
        "0.050000": 0.85,
        ...
    },
    ...
}
```


## Acknowledgements

This code leverages the following repository for impelementation of ST-PPP algorithm:
- [ST-PPP](https://github.com/pbideau/Event-ST-PPP)




