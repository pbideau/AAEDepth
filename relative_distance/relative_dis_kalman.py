import os
import numpy as np
import sys
from rela_utils.visual_util import *
from rela_utils.warp_util import get_warped_cent
from rela_utils.re_utils import *
import argparse
import cProfile
from datetime import datetime
import matplotlib.pyplot as plt

def save_depth_images(gt_relative_depth_img, est_relative_depth_img, filepath,cm = 'jet', clim=None, cb_max=None):
    plt.figure(figsize=(20, 20))
    
    plt.subplot(2, 1, 1)
    plt.title('Ground Truth Relative Depth Image')
    plt.imshow(gt_relative_depth_img, cmap=cm)
    if clim:
        plt.clim(0, clim)
    plt.colorbar()

    plt.subplot(2, 1, 2)
    plt.title('Estimated Relative Depth Image')
    plt.imshow(est_relative_depth_img, cmap=cm)
    if cb_max:
        plt.clim(0, cb_max)
    plt.colorbar()

    plt.savefig(filepath)
    plt.close()


class KF():
    def __init__(self, n):
        self.n = n
        self.I = np.eye(n)
        self.x = None
        self.P = None
        self.F = None
        self.Q = None
        self.H = None

    def predict(self):
        self.x = np.dot(self.F, self.x) 
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

    def update(self, z, R):
        y = z - np.dot(self.H,self.x) 
        PHt = np.dot(self.P, self.H.T)
        S = np.dot(self.H, PHt) + R   
        S_inverse = np.linalg.inv(S) 
        K = np.dot(PHt , S_inverse)

        self.x = self.x + np.dot(K , y)
        self.P = np.dot((self.I - np.dot(K , self.H)) , self.P)

def get_txt_dict(est_dir_our):
    txt_dict = {}
    stppp_files = os.listdir(est_dir_our)
    stppp_txt_files = [file for file in stppp_files if file.endswith('.txt')]    
    for file in stppp_txt_files:
        # print('file',file)
        if 'global' in file:
            continue
        obj_id = int(file.split('_')[1][:8])
        file_path = os.path.join(est_dir_our, file)
        txt_dict[obj_id] = np.loadtxt(file_path)

    return txt_dict

def get_rot_vel(txt_file,t,est_freq_our,object_id):
    if txt_file.ndim == 2: # more than one line
        mid_timestamp = (txt_file[:,1] + txt_file[:,2]) / 2
        if np.min(abs(mid_timestamp - t)) <= 1/est_freq_our: 
            frame_id_ = np.argmin(np.abs(mid_timestamp - t))
            line = txt_file[frame_id_]
        else:
            # Skip this object_id if the timestamp is out of range
            return None
    elif txt_file.ndim == 1:
        print('-------warning!! txt_file.ndim == 1-------')
        # Treat as a single line with at least 3 elements (including two timestamps)
        mid_timestamp = (txt_file[1] + txt_file[2]) / 2
        if abs(mid_timestamp - t) <= 1/est_freq_our: 
            line = txt_file
        else:
            # Skip this object_id if the timestamp is out of range
            return None
    else:
        print('-------warning!! Unexpected txt_file dimensions or missing columns-------')
        print('object_id,txt_file',object_id,txt_file)
        quit()
    rot_vel = line[4:7]
    return rot_vel

def compute_relative_distance(rot_flow_mat_dic,ob1_id,ob2_id):  # # dis(obj1/obj2)
    # print('ob1_id',ob1_id,'ob2_id',ob2_id)
    ob1 = rot_flow_mat_dic[ob1_id]
    ob2 = rot_flow_mat_dic[ob2_id]
    inv_ob1 = np.linalg.pinv(ob1)
    relative_distance = inv_ob1 @ ob2
    relative_distance = relative_distance[0,0]
    return relative_distance

def compute_rot_flow_mat_dic(object_ids,txt_dict,t,est_freq_our,mask,fx,fy,px,py):
    rot_flow_mat_dic = {}
    for object_id in object_ids:
        if object_id not in txt_dict:
            # print('object_id not in txt_dict','object_id',object_id)
            continue
        txt_file = txt_dict[object_id]
        rot_vel = get_rot_vel(txt_file, t,est_freq_our,object_id)
        if rot_vel is None:
            # print('rot_vel is None','object_id',object_id,'object_ids',object_ids,'t',t)
            # quit()
            continue
        mass_y, mass_x  = np.where(mask == object_id)  # this is right order of y,x
        cent_x = np.mean(mass_x)
        cent_y = np.mean(mass_y)   
        warped_x,warped_y = get_warped_cent((cent_x,cent_y),rot_vel,fx,fy,px,py,1/est_freq_our)
        rot_flow = (cent_x - warped_x,cent_y - warped_y)
        rot_flow_numpy =  np.reshape(rot_flow, (2,1))
        rot_flow_mat_dic[object_id] = rot_flow_numpy
    # print('rot_flow_mat_dic',rot_flow_mat_dic)
    return rot_flow_mat_dic

def save_img_npy(t,results_save_path,sequence,est_relative_depth_img,gt_relative_depth_img):
    # Save estimated depth image
    img_save_path = os.path.join(results_save_path, f"{sequence}_est_img_npy")
    if not os.path.exists(img_save_path):
        os.makedirs(img_save_path)
    print('save_est_npy', os.path.join(img_save_path, f"{t}.npy"))
    np.save(os.path.join(img_save_path, f"{t}.npy"), est_relative_depth_img)

    sys.stdout.flush()    

def Euclidean_compute(cent,fx,px,py):
    cent_x, cent_y = cent
    temp = (cent_x-px) ** 2 + (cent_y-py) ** 2 + fx**2
    res = np.sqrt(temp)
    return res

def compute_distance_sequence(dataset_dir,sequence,est_dir_our_base,est_freq_our,
                              fx,fy,px,py,results_save_path,save_npy_flag = 1,save_dicts = 1,
                              plot_rot_flows_flag = 0):
    est_dir_our = os.path.join(est_dir_our_base, sequence)
    timestamps_masks,mask_npz,mask_names,gt_depth_npz,gt_depth_names = parse_gt_data(dataset_dir, sequence)

    result_obj_dict = {}
    result_frame_dict = {}

    kf_dict = {k: KF(1) for k in range(5000, 28001,1000) if k != 22000}
    for kf in kf_dict.values():
        kf.Q = np.array([[Q]])
        kf.F = np.array([[1.]])
        kf.H = np.array([[1.]])

    txt_dict = get_txt_dict(est_dir_our)

    i=0
    while True:
        t = (i + 0.5)/est_freq_our
        # print('i',i)
        i = i + 1
        if t>timestamps_masks[len(timestamps_masks)-1]:
            break   
        
        gt_frame_id = np.argmin(np.abs(np.array(timestamps_masks) - t))
        if i % 50 == 1:
            print('t',t,'gt_frame_id',gt_frame_id)
        mask = mask_npz[mask_names[gt_frame_id]]
        depth = gt_depth_npz[gt_depth_names[gt_frame_id]]
        object_ids = get_object_ids(mask)

        gt_relative_distance_dict,reference_object_id = compute_gt_relative_distance_dict(mask, depth, object_ids)  # find the object has biggest area in the mask to be the referrence obj
        if gt_relative_distance_dict == None:
            continue
        gt_relative_depth_img = np.zeros_like(mask)
        for object_id in gt_relative_distance_dict:
            gt_relative_depth_img = np.where(mask == object_id,gt_relative_distance_dict[object_id],gt_relative_depth_img)
        # plot_one_relative_depth_img(gt_relative_distance_dict, gt_relative_depth_img,gt_frame_id)
        # depth_frame_ts = t
        # plot_mask_depth(gt_frame_id,mask,depth_frame_ts,gt_relative_depth_img,plot_FLAG = 0)  

        rot_flow_mat_dic = compute_rot_flow_mat_dic(object_ids,txt_dict,t,est_freq_our,mask,fx,fy,px,py)  
        if reference_object_id not in list(rot_flow_mat_dic.keys()):
            print('reference_object_id not in list(rot_flow_mat_dic.keys())','reference_object_id',reference_object_id,list(rot_flow_mat_dic.keys()))
            continue

        cent_dict = {} 
        for object_id in rot_flow_mat_dic.keys():
            mass_y, mass_x = np.where(mask == object_id)  # this is the right order of y, x
            cent_x = np.mean(mass_x)
            cent_y = np.mean(mass_y)
            cent_dict[object_id] = (cent_x, cent_y)

        est_relative_distance_dict = {}
        missing_object = 0
        cov_dict = {}
        magnitude_dict = {}
        E2 = Euclidean_compute(cent_dict[reference_object_id],fx,px,py)
        for object_id in kf_dict.keys():
            if object_id in object_ids:
                if object_id in rot_flow_mat_dic.keys():
                    mass_y, mass_x  = np.where(mask == object_id)  # this is right order of y,x
                    cent_x = np.mean(mass_x)
                    cent_y = np.mean(mass_y)   
                    relative_distance = compute_relative_distance(rot_flow_mat_dic, object_id, reference_object_id)

                    E1 = Euclidean_compute(cent_dict[object_id],fx,px,py)
                    rela_depth = relative_distance * E2 / E1

                    magnitude = np.linalg.norm(rot_flow_mat_dic[object_id])
                    c=1
                    R = c*(magnitude**(-a)) 
                    magnitude_dict[object_id] = magnitude
                    # print('R',R,'magnitude',magnitude)
                    if kf_dict[object_id].x == None:
                        kf_dict[object_id].P = np.array(R)
                        kf_dict[object_id].x = np.array(rela_depth)
                    # kf = kf_dict[object_id]
                    kf_dict[object_id].predict()
                    kf_dict[object_id].update(np.array(rela_depth),np.array(R))
                    est_relative_distance_dict[object_id] = kf_dict[object_id].x[0,0]
                    cov_dict[object_id] = kf_dict[object_id].P[0,0]
                    if rela_depth < 0:
                        missing_object = missing_object + 1

                else:
                    missing_object = missing_object + 1

            else: # TODO prediction for all the other obj                
                if kf_dict[object_id].x is not None:  
                    kf_dict[object_id].predict()

        plot_2_depth_img_Flag = 1
        print('plot_2_depth_img_Flag',plot_2_depth_img_Flag)
        if plot_2_depth_img_Flag:
            est_relative_depth_img = np.zeros_like(mask)   
            for object_id in est_relative_distance_dict:
                est_relative_depth_img = np.where(mask == object_id,est_relative_distance_dict[object_id],est_relative_depth_img)
            plot_2_depth_img(gt_relative_depth_img,est_relative_depth_img,t,gt_frame_id,sequence,show = 0,save = 0)
            # min_value = np.min(gt_relative_depth_img)
            min_value = 0
            max_value = np.max(gt_relative_depth_img)
            # print("Minimum Value:", min_value, "Maximum Value:", max_value)

            # gt_depth_img_path = os.path.join(save_dir,current_date,sequence+f'_gt_depth', f'gt_depth_{gt_frame_id:03d}.png')
            est_depth_img_path = os.path.join(results_save_path,sequence, f'est_depth_{gt_frame_id:03d}.png')
            # print('gt_depth_img_path',gt_depth_img_path)
            print('est_depth_img_path',est_depth_img_path)
            # os.makedirs(os.path.dirname(gt_depth_img_path), exist_ok=True)
            os.makedirs(os.path.dirname(est_depth_img_path), exist_ok=True)


    
            # save_depth_images(gt_relative_depth_img, est_relative_depth_img, gt_depth_img_path, clim=0, cb_max=np.max(gt_relative_depth_img))
            save_depth_images(gt_relative_depth_img, est_relative_depth_img, est_depth_img_path, clim=0, cb_max=np.max(gt_relative_depth_img)+0.2)

            # cm = 'jet'
            # plt.imsave(gt_depth_img_path, gt_relative_depth_img, cmap=cm, vmin=min_value, vmax=max_value)
            # plt.imsave(est_depth_img_path, est_relative_depth_img, cmap=cm, vmin=min_value, vmax=max_value)
            # plt.imsave(gt_depth_img_path, gt_relative_depth_img, cmap=cm, vmin=min_value, vmax=max_value, colorbar=True)
            # plt.imsave(est_depth_img_path, est_relative_depth_img, cmap=cm, vmin=min_value, vmax=max_value, colorbar=True)

            # plot_one_relative_depth_img(est_relative_distance_dict ,est_relative_depth_img ,gt_frame_id)
        if plot_rot_flows_flag:
        # if any(value > 2000 for value in est_relative_distance_dict.values())and plot_rot_flows_flag:
        # if negative_obj >0 and plot_rot_flows_flag:
            # print("Warning: One or more values in est_relative_distance_dict are greater than 2000.")
            est_relative_depth_img = np.zeros_like(mask)   
            for object_id in est_relative_distance_dict:
                est_relative_depth_img = np.where(mask == object_id,est_relative_distance_dict[object_id],est_relative_depth_img)
            plot_rot_flows(mask,rot_flow_mat_dic,gt_frame_id,est_relative_distance_dict,est_relative_depth_img,t,sequence,save=0)

        # save the npy imgs 
        # if save_npy_flag:
        #     save_img_npy(t,results_save_path,sequence,est_relative_depth_img,est_relative_depth_img)
        
        if save_dicts:
            result_frame_dict["{:.6f}".format(t).rstrip('0')] = est_relative_distance_dict
            for key, value in est_relative_distance_dict.items():
                if not key in result_obj_dict:
                    result_obj_dict[key] = {'distance': [], 'ts': [],'covariance': [],'magnitude': []}
                result_obj_dict[key]['distance'].append(value)
                result_obj_dict[key]['ts'].append(t)
                result_obj_dict[key]['covariance'].append(cov_dict[key])
                result_obj_dict[key]['magnitude'].append(magnitude_dict[key])
        
             
    if save_dicts:
        # save dicts
        frame_dict_path = f"{results_save_path}/{sequence}_rel_res_frame_dict.npy"
        obj_dict_path = f"{results_save_path}/{sequence}_rel_res_obj_dict.npy"
        np.save(frame_dict_path, result_frame_dict)
        np.save(obj_dict_path, result_obj_dict)
        print(frame_dict_path)
        print(obj_dict_path)
        sys.stdout.flush()    


if __name__ == '__main__':
    profiler = cProfile.Profile()
    profiler.enable()
    parser = argparse.ArgumentParser(description='Code to compute relative distance from st-ppp result')
    parser.add_argument('-d', '--dataset_name', default='sfm', help="Name of dataset, default as 'sfm'")
    parser.add_argument('-s', '--seq', default=2, help="Integer indexing the sequnece array, default as 0")
    parser.add_argument('-sv', '--save_dir', default='relative_output', help="Path to saved output")
    parser.add_argument('-p', '--path', default='dataset', help="Path to dataset")
    parser.add_argument('-e', '--estimate', default='stppp_output', help="Path to stppp result")
    parser.add_argument('-f', '--frequency', default=20, help="frequency of result, default as 20")

    args = parser.parse_args()
    dataset_name = args.dataset_name
    sequence_idx = int(args.seq)
    save_dir = args.save_dir
    dataset_dir = args.path
    est_dir_our_base = args.estimate
    ESTIMATION_FREQ_OUR = int(args.frequency)



    # dataset_name = 'imo'  # sfm   sfm_ll imo
    if dataset_name == 'sfm':
        sequences = ['scene_03_00_000000', 'scene_03_01_000000', 'scene_03_02_000000', 'scene_03_02_000001',
                    'scene_03_02_000002', 'scene_03_02_000003', 'scene_03_03_000000', 'scene_03_03_000001',
                    'scene_03_03_000002', 'scene_03_04_000000']
    elif dataset_name == 'sfm_ll':
        sequences = ['scene17_d_01_000000','scene17_d_02_000000']
    elif dataset_name == 'imo':
        sequences = ['scene13_dyn_test_01_000000','scene13_dyn_test_02_000000','scene13_dyn_test_03_000000','scene14_dyn_test_01_000000']


    current_date = datetime.now().strftime('%Y-%m-%d')

    F_X = 519.638  # in meta
    F_Y = 519.383972
    P_X = 321.661011
    P_Y = 240.727005
    # ESTIMATION_FREQ_OUR = 1 / 0.05  # 20Hz
    ESTIMATION_FREQ_OUR = 20

    
    Q = 0.001
    a = 1
    last_part_of_est_dir_our_base = os.path.basename(est_dir_our_base)
    results_save_path = f"{save_dir}/{current_date}_{dataset_name}_{last_part_of_est_dir_our_base}"
    print('results_save_path',results_save_path)
    if not os.path.exists(results_save_path):
        os.makedirs(results_save_path)
    sequence = sequences[sequence_idx]
    # for sequence in sequences:
    print("computing sequence: ", sequence)
    sys.stdout.flush()
    compute_distance_sequence(dataset_dir,sequence,est_dir_our_base,
                            ESTIMATION_FREQ_OUR,F_X,F_Y,P_X,P_Y,results_save_path, 
                            save_npy_flag = 0,save_dicts = 1,
                            plot_rot_flows_flag = 0)










