import glob
import os
import numpy as np
from matplotlib import pyplot as plt
import sys
parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent)
from rela_utils.re_utils import *
from evaluation.gen_gt_ave_rel_npy import save_gt_dicts

def plot_one_obj_error(object_id,rel_res_obj_dict,rel_gt_obj_dict):
    plt.close()
    est_obj_depths = rel_res_obj_dict[object_id]['distance']
    est_obj_time = rel_res_obj_dict[object_id]['ts']
    est_obj_cov = rel_res_obj_dict.get(object_id, {}).get('covariance', None)
    gt_obj_depths = rel_gt_obj_dict[object_id]['distance']
    gt_obj_time = rel_gt_obj_dict[object_id]['ts']

    tick_start = np.floor(min(min(est_obj_time),min(gt_obj_time)) * 10) / 10
    # tick_start = np.floor(min(min(est_obj_time),min(gt_obj_time)))
    tick_end = np.ceil(max(max(est_obj_time),max(gt_obj_time)) * 10) / 10
    length_xticks = tick_end - tick_start
    tick_interval = 1
    if length_xticks >tick_interval:
        plt.figure(figsize=(int(length_xticks)*2, 5))
        plt.scatter(est_obj_time, est_obj_depths, marker='o', color='blue', label='new')
        plt.scatter(gt_obj_time, gt_obj_depths, marker='o', color='red', label='groundtruth', alpha=0.5)
        # plt.scatter(old_dict[obj]['ts'], old_dict[obj]['distance'], marker='o', color='y',
        #             label='old', s=10, alpha=0.5)
        # plt.errorbar(est_obj_time, est_obj_depths, yerr=est_obj_cov, fmt='o', color='blue', label='new', capsize=5)
        if est_obj_cov is not None:
            est_obj_cov = np.array(est_obj_cov)
            sigma3 = 3 * np.sqrt(est_obj_cov)
            # plt.plot(est_obj_time,est_obj_depths +sigma3, color="orange", label="3-sigma bound, upper", linestyle="dotted")
            # plt.plot(est_obj_time,est_obj_depths -sigma3, color="darkgreen", label="3-sigma bound, lower", linestyle="dotted")
            plt.fill_between(est_obj_time, est_obj_depths + sigma3, est_obj_depths - sigma3, color="lightskyblue", alpha=0.5, label="3-sigma area")
        plt.xlabel('Time')
        plt.ylabel('Relative Distance')
        plt.title(f'Scatter Plot of Relative Distance vs Time, obj{object_id}')
        plt.legend()
        plt.xticks(np.arange(tick_start, tick_end + tick_interval, tick_interval))
        # plt.xticks(np.arange(0.7, 5.5, tick_interval))
        plt.tight_layout()
        # plt.show()
        return 1
    else:
        return 0
    # Show the plot
    # plt.show()

def plot_error(gt_obj_dict_dir,res_obj_dict_dir,sequence,plt_error_dir):
    # old_dict_dir = '/Users/cainan/Desktop/active_perception/event_depth_esti/code/stppp_and_output/x-student-events/output/2024-01-15_oldmethod'
    print('gt_obj_dict_dir',gt_obj_dict_dir)
    
    rel_res_obj_dict_file = glob.glob(os.path.join(res_obj_dict_dir, '*'+sequence+'*'+ 'rel_res_obj_dict' +'*.npy')) 
    rel_gt_obj_dict_file = glob.glob(os.path.join(gt_obj_dict_dir, '*'+sequence+'*'+ 'rel_gt_obj_dict' +'*.npy')) 
    # old_file = glob.glob(os.path.join(old_dict_dir, '*'+sequence+'*'+ 'rel_res_obj_dict' +'*.npy')) 
    if len(rel_res_obj_dict_file) != 1:
        print('!!!!len(depth_files) != 1!!!!!')
        quit()
    if len(rel_gt_obj_dict_file) != 1:
        print('!!!!len(depth_files) != 1!!!!!')
        quit()
    rel_res_obj_dict = np.load(rel_res_obj_dict_file[0], allow_pickle=True).item()
    rel_gt_obj_dict = np.load(rel_gt_obj_dict_file[0], allow_pickle=True).item()
    # old_dict = np.load(old_file[0], allow_pickle=True).item()
    object_ids = list(rel_res_obj_dict.keys())
    object_ids.sort()
    # object_id = 6000
    for object_id in object_ids:
        # # Plot the histogram
        # est_obj_cov = rel_res_obj_dict[object_id]['covariance']
        # plt.figure(figsize=(8, 6))
        # plt.hist(est_obj_cov, bins=10, color='skyblue', edgecolor='black')
        # plt.xlabel('Covariance Values')
        # plt.ylabel('Frequency')
        # plt.title(f'Histogram of Estimation Covariance obj{object_id}')
        # plt.show()

        temp = plot_one_obj_error(object_id,rel_res_obj_dict,rel_gt_obj_dict)
        if temp != 0:
            save_path = f"{plt_error_dir}/{sequence}_{object_id}.png"
            print('save_path',save_path)
            plt.savefig(save_path)

def evaluate(result_our_dict, result_gt_dict,ERROR_THRESHOLD):
    error_dict_key_list = ['error', 'rel_error', 'squared_rel_error', 'log_error', 'delta']
    error_dict = {key:[] for key in error_dict_key_list}
    negative_obj = 0
    mismatch = 0

    for key, value in result_our_dict.items():
        if key in ["missing objects", "total objects"]:
            continue

        est_obj_t = result_our_dict[key]['ts']
        est_obj_dist = result_our_dict[key]['distance']

        gt_obj_t = result_gt_dict[key]['ts']
        gt_obj_t = np.array(gt_obj_t)
        gt_obj_dist = result_gt_dict[key]['distance']
        for t, est in zip(est_obj_t, est_obj_dist):
            if est < 0:
                negative_obj += 1
                continue
            if np.min(np.abs(gt_obj_t - t)) >= ERROR_THRESHOLD:
                mismatch += 1
                continue

            idx = np.argmin(np.abs(np.array(gt_obj_t) - t))
            gt = gt_obj_dist[idx]
            
            error = gt - est
            abs_error = np.abs(error)
            rel_error = abs_error / gt
            squared_rel_error = abs_error * abs_error / gt
            log_error = np.log(gt) - np.log(est)
            delta = max(est/gt,gt/est)

            error_dict['error'].append(abs_error)
            error_dict['rel_error'].append(rel_error)
            error_dict['squared_rel_error'].append(squared_rel_error)
            error_dict['log_error'].append(log_error)
            error_dict['delta'].append(delta)
    delta_array = np.array(error_dict['delta'])
    statistics = {'mean_error': np.mean(error_dict['error']),
        'mean_std': np.std(error_dict['error']),
        'mean_rel_error': np.mean(error_dict['rel_error']),
        'mean_squared_rel_error': np.mean(error_dict['squared_rel_error']),
        'RMSE_linear': np.sqrt(np.mean(np.square(error_dict['error']))),
        'RMSE_log': np.sqrt(np.mean(np.square(np.abs(error_dict['log_error'])))),
        'RMSE_log_scale_invariant': np.mean(np.square(error_dict['log_error'] + np.mean(error_dict['log_error']))),
        'percentage_gt_1_25': np.sum(delta_array < 1.25) / len(error_dict['delta']) * 100,
        'percentage_gt_1_25_sq': np.sum(delta_array< 1.25**2) / len(error_dict['delta']) * 100,
        'percentage_gt_1_25_cube': np.sum(delta_array < 1.25**3) / len(error_dict['delta']) * 100,
        'negative_obj':negative_obj,
        'mismatch':mismatch
    }   
    return statistics

def evaluate_dataset(sequences,res_obj_dict_dir,gt_obj_dict_dir,ERROR_THRESHOLD):
    statistics_dict_key_list = ['mean_error','mean_std','mean_rel_error','mean_squared_rel_error',
                                'RMSE_linear','RMSE_log','RMSE_log_scale_invariant',
                                'percentage_gt_1_25','percentage_gt_1_25_sq','percentage_gt_1_25_cube',
                                'negative_obj','mismatch']
    statistics_dict = {key: [] for key in statistics_dict_key_list}

    for sequence in sequences:
        # print('sequence',sequence)
        rel_res_obj_dict_file = glob.glob(os.path.join(res_obj_dict_dir, '*'+sequence+'*'+ 'rel_res_obj_dict' +'*.npy')) 
        rel_gt_obj_dict_file = glob.glob(os.path.join(gt_obj_dict_dir, '*'+sequence+'*'+ 'rel_gt_obj_dict' +'*.npy'))         

        if len(rel_res_obj_dict_file) != 1:
            print('!!!!len(rel_res_obj_dict_file) != 1!!!!!')
            print('res_obj_dict_dir',res_obj_dict_dir,'\nrel_res_obj_dict_file',rel_res_obj_dict_file)
            return
        if len(rel_gt_obj_dict_file) != 1:
            print('!!!!len(rel_gt_obj_dict_file) != 1!!!!!')
            return
        rel_res_obj_dict = np.load(rel_res_obj_dict_file[0], allow_pickle=True).item()
        rel_gt_obj_dict = np.load(rel_gt_obj_dict_file[0], allow_pickle=True).item()
    
        statistics = evaluate(rel_res_obj_dict, rel_gt_obj_dict,ERROR_THRESHOLD)
        # print('statistics',statistics)
        for key in statistics_dict.keys():
            statistics_dict[key].append(statistics[key])

    # print(f'statistics_dict\n{statistics_dict}')  # results of all sequences
    statistics_dict_mean = {key:np.mean(statistics_dict[key]) for key in statistics_dict.keys() if key not in ['negative_obj', 'mismatch']}

    if mean_table ==0:
        print_dict = {"Mean of mean error:": statistics_dict_key_list[0],
                      "Mean of std:": statistics_dict_key_list[1]}
    else:
        print_dict = {
            'Abs Relative difference ARD:': statistics_dict_key_list[2],
            'Squared Relative difference SRD:': statistics_dict_key_list[3],
            'Mean of mean RMSE_linear:': statistics_dict_key_list[4],
            'Mean of mean RMSE_log:': statistics_dict_key_list[5],
            'Mean of mean RMSE_log_scale_invariant:': statistics_dict_key_list[6],
            'Mean of delta<1.25%:': statistics_dict_key_list[7],
            'Mean of delta<1.25^2%:': statistics_dict_key_list[8],
            'Mean of delta<1.25^3%:': statistics_dict_key_list[9],
        }
        
    for title, statistics_mean_key in print_dict.items():
        print(title,statistics_dict_mean[statistics_mean_key])

    if mean_table == 1:
        print("& {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} \\\\".format(
            statistics_dict_mean['RMSE_linear'],
            statistics_dict_mean['RMSE_log'],
            statistics_dict_mean['RMSE_log_scale_invariant'],
            statistics_dict_mean['mean_rel_error'],
            statistics_dict_mean['mean_squared_rel_error'],
            statistics_dict_mean['percentage_gt_1_25'],
            statistics_dict_mean['percentage_gt_1_25_sq'],
            statistics_dict_mean['percentage_gt_1_25_cube']
        ))

    print(res_obj_dict_dir + "\n")

if __name__ == '__main__':
    dataset_name = 'sfm'  #  sfm_ll  sfm
    mean_table = 1 # mean 0 table 1
    if dataset_name == 'sfm':
        sequences = ['scene_03_00_000000', 'scene_03_01_000000', 'scene_03_02_000000', 'scene_03_02_000001',
                    'scene_03_02_000002', 'scene_03_02_000003', 'scene_03_03_000000', 'scene_03_03_000001',
                    'scene_03_03_000002', 'scene_03_04_000000']
        
        # sequences = ['scene_03_02_000000',
        #             'scene_03_02_000002', 'scene_03_02_000003'
        #             ,'scene_03_03_000000']#,'scene_03_02_000001']

        # sequences = ['scene_03_02_000000',
        #             'scene_03_02_000002', 'scene_03_02_000003'
        #             ,'scene_03_03_000000','scene_03_02_000001','scene_03_00_000000','scene_03_01_000000']
        dataset_path = '/Users/cainan/Desktop/active_perception/semester_poster/EVIMO/data/evimo_npz/samsung_mono/sfm/eval'
        dir = '/Users/cainan/Desktop/active_perception/event_depth_esti/code/stppp_and_output/x-student-events/output'
        # list_res_obj_dict_dir = glob.glob(os.path.join(dir, '*'+ '2024-02-29' +'*'+'a2_b0_'+ '*' +'*'+'02'))
        # list_res_obj_dict_dir = glob.glob(os.path.join(dir,'mar', '*'+ '2024-03-07' +'*'+'a2_b0_'+ '*' +'*'+'_1_50hz'))
        # list_res_obj_dict_dir = glob.glob(os.path.join(dir,'mar', '*'+ '2024-03-06' +'*'+'a2_b0_'+ '*' +'*'+'_1'))
        list_res_obj_dict_dir = glob.glob(os.path.join(dir,'mar', '*'+ '2024-03-21' +'*'+'tor'+ '*' +'*'+'_1'))
        print('list_res_obj_dict_dir',list_res_obj_dict_dir)
        # list_res_obj_dict_dir = []
        # res_obj_dict_dir = '/Users/cainan/Desktop/active_perception/event_depth_esti/code/stppp_and_output/x-student-events/output/mar/2024-03-07measureNoise_inversepower_a2_b0_cov_Q0.001_depth_sfm_t_050_050_1'
        # res_obj_dict_dir = '/Users/cainan/Desktop/active_perception/event_depth_esti/code/e2depth/e2depth_output/2024-02-13_e2depth_evimo_average_rela_sfm'
        # list_res_obj_dict_dir.append(res_obj_dict_dir)
        # res_obj_dict_dir = '/Users/cainan/Desktop/active_perception/event_depth_esti/code/emvs/emvs_output/est_average_depth_img/2024-02-13_EMVS_evimo_average_rela_sfm1s'
        # list_res_obj_dict_dir.append(res_obj_dict_dir)


    elif dataset_name == 'sfm_ll':
        # sequences = ['scene17_d_01_000000', 'static_objects_dark_00_000003', 'static_objects_dark_00_000002', 
        #             'static_objects_dark_00_000000', 'static_objects_dark_00_000001', 'scene17_d_02_000000']
        sequences = ['scene17_d_01_000000','scene17_d_02_000000']
        # sequences = sequences[1:]
        dataset_path = '/Users/cainan/Desktop/active_perception/semester_poster/EVIMO/data/evimo_npz/samsung_mono/sfm_ll/eval'

        dir = '/Users/cainan/Desktop/active_perception/event_depth_esti/code/stppp_and_output/x-student-events/output'
        list_res_obj_dict_dir = glob.glob(os.path.join(dir, '*'+ '2024-03-07' +'*'+'a2_b0_'+ '*' +'sfm_ll'+'*' + ''+ 'gpu'))
        print('list_res_obj_dict_dir',list_res_obj_dict_dir)

        # list_res_obj_dict_dir = []
        # res_obj_dict_dir = '/Users/cainan/Desktop/active_perception/event_depth_esti/code/e2depth/e2depth_output/2024-02-13_e2depth_evimo_average_rela_sfm_ll'
        # list_res_obj_dict_dir.append(res_obj_dict_dir)
        # # res_obj_dict_dir = '/Users/cainan/Desktop/active_perception/event_depth_esti/code/emvs/emvs_output/est_average_depth_img/2024-02-13_EMVS_evimo_average_rela_ll2s'
        # res_obj_dict_dir = '/Users/cainan/Desktop/active_perception/event_depth_esti/code/emvs/emvs_output/est_average_depth_img/2024-02-13_EMVS_evimo_average_rela_sfm_ll1s'
        # list_res_obj_dict_dir.append(res_obj_dict_dir)




    if dataset_name == 'sfm':
        # gt_obj_dict_dir = 'output/2024-02-07_rel_gt_obj_dict'
        gt_obj_dict_dir = '/Users/cainan/Desktop/active_perception/event_depth_esti/code/stppp_and_output/x-student-events/output/2024-02-12_rel_gt_obj_dict_sfm'
        # gt_obj_dict_dir = '/Users/cainan/Desktop/active_perception/event_depth_esti/code/stppp_and_output/x-student-events/output/2024-03-02_rel_gt_obj_dict_sfm'
    elif dataset_name == 'sfm_ll':
        gt_obj_dict_dir = '/Users/cainan/Desktop/active_perception/event_depth_esti/code/stppp_and_output/x-student-events/output/2024-02-13_rel_gt_obj_dict_sfm_ll'
        # gt_obj_dict_dir = '/Users/cainan/Desktop/active_perception/event_depth_esti/code/stppp_and_output/x-student-events/output/2024-03-02_rel_gt_obj_dict_sfm_ll'
    evimo2_THRESHOLD = 1/40  # frequency of evimo2 40Hz
    stppp_THRESHOLD = 0.05
    # stppp_THRESHOLD = 1/10
    ERROR_THRESHOLD = min(evimo2_THRESHOLD,stppp_THRESHOLD)

    # print(res_obj_dict_dir)
    # evaluate_dataset(sequences,res_obj_dict_dir,gt_obj_dict_dir,ERROR_THRESHOLD)



    for res_obj_dict_dir in list_res_obj_dict_dir:
        evaluate_dataset(sequences,res_obj_dict_dir,gt_obj_dict_dir,ERROR_THRESHOLD)


        # !!!!!!plot error part!!!
        # plt_error_dir = res_obj_dict_dir
        # if not os.path.exists(plt_error_dir):
        #     os.makedirs(plt_error_dir)

        # for sequence in sequences:
        #     print('sequence',sequence)
        #     plot_error(gt_obj_dict_dir,res_obj_dict_dir,sequence,plt_error_dir)

