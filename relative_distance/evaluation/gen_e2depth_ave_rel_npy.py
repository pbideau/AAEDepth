import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime
import shutil
import glob
import sys
parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent)
# i cannot use relative path
# sys.path.append('/Users/cainan/Desktop/active_perception/event_depth_esti/code/stppp_and_output/x-student-events/relative_distance')
# # print('parent',parent)
from rela_utils.re_utils import *
from rela_utils.visual_util import *

def load_e2depth_predictions(predictions_evimo, sequence):
    prediction_files = sorted(glob.glob(os.path.join(predictions_evimo, sequence, 'data', 'depth*.npy')))
    print("Number of prediction files:", len(prediction_files))
    if not prediction_files:
        print('No prediction files found!')
        print(predictions_evimo)
        quit()
    prediction_files.sort()

    # print("predictions_evimo + sequence", os.path.join(predictions_evimo, sequence))
    boundary_timestamps_path = glob.glob(os.path.join(predictions_evimo, '*' + sequence + '*', 'boundary_timestamps.txt'))
    if len(boundary_timestamps_path) != 1:
        print('!!!!len(boundary_timestamps_path) != 1!!!!!')
        print('boundary_timestamps_path', boundary_timestamps_path)
        quit()
    boundary_timestamps_path = boundary_timestamps_path[0]

    if os.path.exists(boundary_timestamps_path):  # Check if file exists before attempting to load
        boundary_timestamps = np.loadtxt(boundary_timestamps_path)
    else:
        print("No boundary_timestamps.txt file found for this sequence.")
        quit() 

    start_timestamps = boundary_timestamps[:, 1]
    end_timestamps = boundary_timestamps[:, 2]
    # print(f'len(start_timestamps) {len(start_timestamps)}   len(prediction_files) {len(prediction_files)} ')

    if len(start_timestamps) != len(prediction_files):
        raise ValueError("The sizes of boundary_timestamps.txt and dataset_events_xy are not consistent.")

    # print("Path to predictions dataset:", os.path.join(predictions_evimo, sequence))
    # print("Path to boundary timestamps file:", boundary_timestamps_path)
    return prediction_files,start_timestamps,end_timestamps,boundary_timestamps_path

def recover_predicted_depth(prediction, clip_distance, reg_factor=3.70378):
    # normalize prediction (0 - 1)
    prediction = np.exp(reg_factor * (prediction - np.ones((prediction.shape[0], prediction.shape[1]), dtype=np.float32)))
    prediction *= clip_distance

    return prediction

def inv_depth_to_depth(prediction, reg_factor=3.70378):
    prediction = np.exp(reg_factor * (prediction - np.ones((prediction.shape[0], prediction.shape[1]), dtype=np.float32)))
    prediction = 1/prediction
    prediction = prediction/np.amax(prediction)
    prediction = np.ones((prediction.shape[0], prediction.shape[1]), dtype=np.float32) + np.log(prediction)/reg_factor
    return prediction

def gen_e2depth_ave_rel_npy(sequence,dataset_path,save_dir,predictions_evimo):
    prediction_files,start_timestamps,end_timestamps,boundary_timestamps_path = load_e2depth_predictions(predictions_evimo, sequence)
    masks_timestamps,masks_names,masks,t_ref_events = parse_evimo_masks(dataset_path,sequence)
    current_date = datetime.now().strftime('%Y-%m-%d')    
    dataset_name = dataset_path.split('/')[-2] 
    save_path = os.path.join(save_dir, current_date + f'_e2depth_evimo_average_rela_{dataset_name}', sequence)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    result_obj_dict = {}
    result_frame_dict = {}
    # gt_obj_dict = {}
    # i = 0
    for i in range(200,len(start_timestamps)):
    # while True:
        depth_frame_id = i
        depth_frame_ts = 0.5 * (start_timestamps[i] + end_timestamps[i])
        # print('frame id',i)
        # print(f'ts {ts}')
        print('depth_frame_ts',depth_frame_ts,'end_timestamps[len(end_timestamps)-1]',end_timestamps[len(end_timestamps)-1])
        if depth_frame_ts>end_timestamps[len(end_timestamps)-1]:
            break
        predicted_depth = np.load(prediction_files[i])
        clip_distance = 80.0
        predicted_depth = recover_predicted_depth(predicted_depth, clip_distance)
        # print('np.min(predicted_depth)',np.min(predicted_depth))
        # inv = 0
        # if inv:
        #     print('inv')
        #     predicted_depth = inv_depth_to_depth(predicted_depth)
        mask,mask_frame_id,mask_timestamp,mask_frame_id = find_corresponding_mask(masks_timestamps,masks_names,masks,depth_frame_id,depth_frame_ts)   
        plot_mask_depth(mask_frame_id,mask,depth_frame_ts,predicted_depth,plot_FLAG = 1)

        object_ids = get_object_ids(mask)
        if len(object_ids) == 0:
            print('Warning: len(object_ids) == 0. Skipping this frame.')
            continue
        avg_obj_depth_dict = compute_avg_obj_depth_dict(mask, predicted_depth, object_ids,show = 0)
        if not avg_obj_depth_dict:  # Check if avg_obj_depth_dict is empty
            print("Warning: No objects found. Skipping this frame.")
            continue  # Skip the rest of the loop for this iteration

        reference_object_id = max(avg_obj_depth_dict.keys(), key=lambda obj_id: np.sum(mask == obj_id))
        relative_avg_obj_depth_dict = {key: value / avg_obj_depth_dict[reference_object_id] for key, value in avg_obj_depth_dict.items()}
        
        # !!!! average distance
        # est_average_depth_img = np.zeros_like(mask)
        # for object_id in avg_obj_depth_dict:
        #     est_average_depth_img = np.where(mask == object_id,avg_obj_depth_dict[object_id],est_average_depth_img)
        # e2depth_plot(est_average_depth_img,depth_frame_id,avg_obj_depth_dict,show = 0)
        
        # !!!! relative distance
        est_relative_average_depth_img = np.zeros_like(mask)
        for object_id in relative_avg_obj_depth_dict:
            est_relative_average_depth_img = np.where(mask == object_id,relative_avg_obj_depth_dict[object_id],est_relative_average_depth_img)
        e2depth_plot(est_relative_average_depth_img,depth_frame_id,relative_avg_obj_depth_dict,show = 0)

        save_npy = 0
        if save_npy:
            save_file_path = os.path.join(save_path, "%.09f.npy" % depth_frame_ts)  # "event_tensor_%010d.npy" % i
            np.save(save_file_path, est_relative_average_depth_img) 
            print(f"File saved at: {save_file_path}")
            shutil.copyfile(boundary_timestamps_path,
                            save_path + '/boundary_timestamps.txt')    
        

        result_frame_dict["{:.6f}å".format(depth_frame_ts).rstrip('0')] = relative_avg_obj_depth_dict
        for key, value in relative_avg_obj_depth_dict.items():
            if not key in result_obj_dict:
                result_obj_dict[key] = {'distance': [], 'ts': []}
            result_obj_dict[key]['distance'].append(value)
            result_obj_dict[key]['ts'].append(depth_frame_ts-t_ref_events)

        # i = i +1
    frame_dict_path = f"{save_path}_rel_res_frame_dict.npy"
    obj_dict_path = f"{save_path}_rel_res_obj_dict.npy"
    np.save(frame_dict_path, result_frame_dict)
    np.save(obj_dict_path, result_obj_dict)
    print(frame_dict_path)
    print(obj_dict_path)
    sys.stdout.flush()  
     

if __name__ == '__main__':   
    dataset_name = 'sfm'
    # dataset_name = 'sfm_ll'
    if dataset_name == 'sfm':
        sequences = ['scene_03_00_000000', 'scene_03_01_000000', 'scene_03_02_000000', 'scene_03_02_000001',
                    'scene_03_02_000002', 'scene_03_02_000003', 'scene_03_03_000000', 'scene_03_03_000001',
                    'scene_03_03_000002', 'scene_03_04_000000']
        # lab computer
        # dataset_path = '/home/ncai/code/e2depth/npz_samsung_mono_sfm/samsung_mono/sfm/eval'  # lab computer
        # save_dir = '/home/ncai/code/e2depth/e2depth_evimo_output/'

        dataset_path = '/Users/cainan/Desktop/active_perception/semester_poster/EVIMO/data/evimo_npz/samsung_mono/sfm/eval'
        save_dir = '/Users/cainan/Desktop/active_perception/event_depth_esti/code/e2depth/e2depth_output'
        predictions_evimo_e2depth_dir = '/Users/cainan/Desktop/active_perception/event_depth_esti/code/e2depth/e2depth_output/11_11_evimo_all'
    elif dataset_name == 'sfm_ll':
        sequences = ['scene17_d_01_000000', 'static_objects_dark_00_000003', 'static_objects_dark_00_000002', 
                    'static_objects_dark_00_000000', 'static_objects_dark_00_000001', 'scene17_d_02_000000']
        dataset_path = '/Users/cainan/Desktop/active_perception/semester_poster/EVIMO/data/evimo_npz/samsung_mono/sfm_ll/eval'
        save_dir = '/Users/cainan/Desktop/active_perception/event_depth_esti/code/e2depth/e2depth_output'
        
        predictions_evimo_e2depth_dir="/Users/cainan/Desktop/active_perception/event_depth_esti/code/e2depth/e2depth_output/28_01_evimo_lowlight"
    
    # sequence = sequences[0]
    for sequence in sequences[:]:
        gen_e2depth_ave_rel_npy(sequence,dataset_path,save_dir,predictions_evimo_e2depth_dir)
    print('save_dir',save_dir)


  
