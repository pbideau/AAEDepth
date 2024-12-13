# import glob
import os
import numpy as np
# from matplotlib import pyplot as plt

import sys
parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent)
from rela_utils.re_utils import *

def save_gt_dict(sequence,gt_obj_dict_dir,dataset_path):
    timestamps_masks,mask_npz,mask_names,gt_depth_npz,gt_depth_names = parse_gt_data(dataset_path, sequence)
    gt_obj_dict = {}
    gt_frame_dict = {}
    for i in range(len(timestamps_masks)):
        mask = mask_npz[mask_names[i]]
        mask_ts = timestamps_masks[i]
        if i % 50 == 1:
            print('i',i,'mask_ts',mask_ts)
        depth = gt_depth_npz[gt_depth_names[i]]
        object_ids = get_object_ids(mask)
        if len(object_ids) == 0:
            print('Warning: len(object_ids) == 0. Skipping this frame.')
            continue
        # gt_relative_distance_dict,reference_object_id = compute_gt_relative_distance_dict(mask, depth, object_ids) 
        avg_obj_depth_dict = compute_avg_obj_depth_dict(mask, depth, object_ids,show = 0)
        size_current = 0
        reference_object_id = 0
        for obj_id in avg_obj_depth_dict.keys():
            mask_obj = mask == obj_id
            if np.sum(mask_obj) > size_current:
                size_current = np.sum(mask_obj)
                reference_object_id = obj_id
        gt_relative_distance_dict = {key: value / avg_obj_depth_dict[reference_object_id] for key, value in avg_obj_depth_dict.items()}

        gt_frame_dict["{:.6f}".format(mask_ts).rstrip('0')] = gt_relative_distance_dict

        for key, value in gt_relative_distance_dict.items():
            if not key in gt_obj_dict:
                gt_obj_dict[key] = {'distance': [], 'ts': [],'absolute':[]}
            gt_obj_dict[key]['distance'].append(value)
            gt_obj_dict[key]['ts'].append(timestamps_masks[i])
            gt_obj_dict[key]['absolute'].append(avg_obj_depth_dict[key])

    gt_obj_dict_path = f"{gt_obj_dict_dir}/{sequence}_rel_gt_obj_dict.npy"
    np.save(gt_obj_dict_path, gt_obj_dict)
    print('gt_obj_dict_path',gt_obj_dict_path)

    frame_dict_path = f"{gt_obj_dict_dir}/{sequence}_rel_res_frame_dict.npy"
    np.save(frame_dict_path, gt_frame_dict)
    print('frame_dict_path',frame_dict_path)

    sys.stdout.flush()    

def save_gt_dicts(save_dir,dataset_path,sequences):
    current_date = datetime.now().strftime('%Y-%m-%d')
    dataset_name = dataset_path.split('/')[-2]  
    gt_obj_dict_dir = f"{save_dir}/{current_date}_rel_gt_obj_dict_{dataset_name}"
    # gt_obj_dict_dir = f"{save_dir}/{current_date}_rel_gt_obj_dict"
    print('gt_obj_dict_dir',gt_obj_dict_dir)
    if not os.path.exists(gt_obj_dict_dir):
        os.makedirs(gt_obj_dict_dir)
    for sequence in sequences:
        save_gt_dict(sequence,gt_obj_dict_dir,dataset_path)
    return gt_obj_dict_dir

if __name__ == '__main__':
    dataset_name = 'sfm'  # sfm    sfm_ll
    if dataset_name == 'sfm':
        sequences = ['scene_03_00_000000', 'scene_03_01_000000', 'scene_03_02_000000', 'scene_03_02_000001',
                    'scene_03_02_000002', 'scene_03_02_000003', 'scene_03_03_000000', 'scene_03_03_000001',
                    'scene_03_03_000002', 'scene_03_04_000000']

        dataset_path = '/Users/cainan/Desktop/active_perception/semester_poster/EVIMO/data/evimo_npz/samsung_mono/sfm/eval'

    elif dataset_name == 'sfm_ll':
        sequences = ['scene17_d_01_000000', 'static_objects_dark_00_000003', 'static_objects_dark_00_000002', 
                     'static_objects_dark_00_000000', 'static_objects_dark_00_000001', 'scene17_d_02_000000']
        dataset_path = '/Users/cainan/Desktop/active_perception/semester_poster/EVIMO/data/evimo_npz/samsung_mono/sfm_ll/eval'

    save_dir = 'output'
    gt_obj_dict_dir = save_gt_dicts(save_dir,dataset_path,sequences)
    print('gt_obj_dict_dir',gt_obj_dict_dir)
