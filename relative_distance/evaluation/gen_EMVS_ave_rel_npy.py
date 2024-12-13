from datetime import datetime
import os
import glob

import sys
parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent)
from rela_utils.re_utils import *
from rela_utils.visual_util import *

def gen_EMVS_ave_rel_npy(dataset_path,sequence,predictions_evimo_emvs_dir,save_dir,dataset_name):

    masks_timestamps,masks_names,masks,t_ref_events = parse_evimo_masks(dataset_path,sequence)
    # timestamps_masks,mask_npz,mask_names,gt_depth_npz,gt_depth_names = parse_gt_data(dataset_path, sequence)
        
    gt_depth_npz = np.load(os.path.join(dataset_path, sequence, 'dataset_depth.npz'))
    gt_depth_names = list(gt_depth_npz.keys())
    gt_depth_names.sort()

    # load EMVS estimations
    depth_files = glob.glob(os.path.join(predictions_evimo_emvs_dir, '*'+sequence+'*', '*.npy'))     
    # print(f'depth_files {depth_files}')
    if len(depth_files) == 0:
        print('!!!!len(depth_files) == 0!!!!!')
        quit()
    sequence_folder_name = depth_files[0].split('/')[-2]

    depth_frame_timestamps = np.array([path.split('/')[-1].rsplit('.', 1)[0] for path in depth_files])
    depth_frame_timestamps_float = depth_frame_timestamps.astype(np.float64)
    depth_frame_timestamps_float.sort()

    result_obj_dict = {}
    result_frame_dict = {}
    current_date = datetime.now().strftime('%Y-%m-%d')    
    save_path = os.path.join(save_dir, current_date + f'_EMVS_evimo_average_rela_{dataset_name}', sequence)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    for depth_frame_id in range(len(depth_frame_timestamps_float)):
        depth_frame_ts = depth_frame_timestamps_float[depth_frame_id]
        print(f'depth_frame_ts {depth_frame_ts}')
        temp = str(int(depth_frame_ts)) if depth_frame_ts.is_integer() else str(depth_frame_ts)
        print('temp',temp)
        depth_img = np.load(os.path.join(predictions_evimo_emvs_dir, sequence_folder_name, temp+'.npy'))
        # frame_id = depth_frame_id

        mask,mask_frame_id,mask_timestamp,gt_frame_id = find_corresponding_mask(masks_timestamps,masks_names,masks,depth_frame_id,depth_frame_ts)   
        depth = gt_depth_npz[gt_depth_names[gt_frame_id]]
        # if depth_frame_ts >23:
        overlay_mask_on_depthmap(mask_frame_id,mask,depth_frame_ts, depth_img,plot_FLAG = 1)  
        plot_mask_depth(mask_timestamp,mask,depth_frame_ts,depth_img,plot_FLAG = 0)

        object_ids = get_object_ids(mask)
        if len(object_ids) == 0:
            print('Warning: len(object_ids) == 0. Skipping this frame.')
            continue


        # avg_obj_depth_dict = compute_avg_obj_depth_dict(mask, depth_img, object_ids,show = 0)
        avg_obj_depth_dict = compute_avg_obj_depth_dict_EMVS(mask, depth_img, object_ids,show = 0)
        if not avg_obj_depth_dict:  # Check if avg_obj_depth_dict is empty
            print("Warning: No objects found. Skipping this frame.")
            continue  # Skip the rest of the loop for this iteration

        reference_object_id = max(avg_obj_depth_dict.keys(), key=lambda obj_id: np.sum(mask == obj_id)) 
        # also can check the length of avg_obj_depth_dict.keys and object_ids is the same or not
        relative_avg_obj_depth_dict = {key: value / avg_obj_depth_dict[reference_object_id] for key, value in avg_obj_depth_dict.items()}

        est_relative_average_depth_img = np.zeros_like(mask)
        for object_id in relative_avg_obj_depth_dict:
            est_relative_average_depth_img = np.where(mask == object_id,relative_avg_obj_depth_dict[object_id],est_relative_average_depth_img)
        e2depth_plot(est_relative_average_depth_img,depth_frame_id,relative_avg_obj_depth_dict,show = 0)
        
        
        gt_relative_distance_dict,reference_object_id = compute_gt_relative_distance_dict(mask, depth, object_ids)  # find the object has biggest area in the mask to be the referrence obj
        gt_relative_depth_img = np.zeros_like(mask)
        for object_id in gt_relative_distance_dict:
            # if object_id ==24000:# in txt_dict.keys():
            gt_relative_depth_img = np.where(mask == object_id,gt_relative_distance_dict[object_id],gt_relative_depth_img)
        
        # plot_2_depth_img(gt_relative_depth_img,est_relative_average_depth_img,depth_frame_ts,mask_timestamp,sequence,show = 0,save = 0)

        # TODO save npy
        save_file_path = os.path.join(save_path, "%.05f.npy" % depth_frame_ts)  # "event_tensor_%010d.npy" % i
        np.save(save_file_path, est_relative_average_depth_img) 
        print(f"File saved at: {save_file_path}")

        result_frame_dict["{:.6f}å".format(depth_frame_ts).rstrip('0')] = relative_avg_obj_depth_dict
        for key, value in relative_avg_obj_depth_dict.items():
            if not key in result_obj_dict:
                result_obj_dict[key] = {'distance': [], 'ts': []}
            result_obj_dict[key]['distance'].append(value)
            result_obj_dict[key]['ts'].append(depth_frame_ts-t_ref_events)

    frame_dict_path = f"{save_path}_rel_res_frame_dict.npy"
    obj_dict_path = f"{save_path}_rel_res_obj_dict.npy"
    np.save(frame_dict_path, result_frame_dict)
    np.save(obj_dict_path, result_obj_dict)
    print(frame_dict_path)
    print(obj_dict_path)
    sys.stdout.flush()  

if __name__ == '__main__': 


    dataset_name = 'sfm'
    if dataset_name == 'sfm':
        sequences = ['scene_03_00_000000', 'scene_03_01_000000', 'scene_03_02_000000', 'scene_03_02_000001',
                    'scene_03_02_000002', 'scene_03_02_000003', 'scene_03_03_000000', 'scene_03_03_000001',
                    'scene_03_03_000002', 'scene_03_04_000000']
        dataset_path = '/Users/cainan/Desktop/active_perception/semester_poster/EVIMO/data/evimo_npz/samsung_mono/sfm/eval'
        save_dir = '/Users/cainan/Desktop/active_perception/event_depth_esti/code/emvs/emvs_output/est_average_depth_img'
        # predictions_evimo_emvs_dir = '/Users/cainan/Desktop/active_perception/event_depth_esti/code/emvs/emvs_output/emvs_evimo_output/11_16/evimo_1s'
        predictions_evimo_emvs_dir = '/Users/cainan/Desktop/active_perception/event_depth_esti/code/emvs/emvs_output/emvs_evimo_output/11_12_1_2s/output2s'
    # depth_dir = '/Users/cainan/Desktop/active_perception/event_depth_esti/code/emvs/emvs_output/emvs_evimo_output/11_16/evimo_1s'

    elif dataset_name == 'sfm_ll':
        sequences = ['scene17_d_01_000000', 'static_objects_dark_00_000003', 'static_objects_dark_00_000002', 
                    'static_objects_dark_00_000000', 'static_objects_dark_00_000001', 'scene17_d_02_000000']
        dataset_path = '/Users/cainan/Desktop/active_perception/semester_poster/EVIMO/data/evimo_npz/samsung_mono/sfm_ll/eval'
        # save_dir = '/Users/cainan/Desktop/active_perception/event_depth_esti/code/e2depth/e2depth_output'
        # save_dir = '/Users/cainan/Desktop/active_perception/event_depth_esti/code/evaluation_output/ave_rel'
        save_dir = '/Users/cainan/Desktop/active_perception/event_depth_esti/code/emvs/emvs_output/est_average_depth_img'
        # predictions_evimo_emvs_dir = '/Users/cainan/Desktop/active_perception/event_depth_esti/code/emvs/emvs_output/emvs_evimo_output/output_28_01/output_28_01_emvs_evimo_lowlight_1s'
        predictions_evimo_emvs_dir = '/Users/cainan/Desktop/active_perception/event_depth_esti/code/emvs/emvs_output/emvs_evimo_output/output_28_01/output_28_01_emvs_evimo_lowlight_2s'
    
    # sequence = sequences[0]
    for sequence in sequences[:]:
        gen_EMVS_ave_rel_npy(dataset_path,sequence,predictions_evimo_emvs_dir,save_dir,dataset_name)
    print('save_dir',save_dir)
    
    print('end')
