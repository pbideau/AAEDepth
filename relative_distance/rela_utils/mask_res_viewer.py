import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import glob
import sys

def get_frame_by_index(frames, index):
    frames_names = list(frames.keys())
    frames_names.sort()
    frame_name = frames_names[index]
    return np.copy(frames[frame_name]) # To extract and keep in RAM

def plot_mask_test():  
    print('ploting masks...')
    dataset_dir = '/Users/cainan/Desktop/active_perception/EVIMO/data/samsung_mono/sfm/eval'
    sequences = ['scene_03_00_000000', 'scene_03_01_000000', 'scene_03_02_000000', 'scene_03_02_000001',
                'scene_03_02_000002', 'scene_03_02_000003', 'scene_03_03_000000', 'scene_03_03_000001',
                'scene_03_03_000002', 'scene_03_04_000000']
    sequence = sequences[0]    
    mask_frame_id = 673
    save_name = None

    masks = np.load(os.path.join(dataset_dir + '/' + sequence, 'dataset_mask.npz'))
    mask_img = get_frame_by_index(masks,mask_frame_id)
    colors = ['xkcd:light grey',
              'xkcd:salmon',
              'xkcd:plum',
              'xkcd:lightblue',
              'xkcd:gold',
              'xkcd:darkblue',
              'xkcd:white']
    cm = LinearSegmentedColormap.from_list('foo', colors[0:29], 29)
    im = plt.imshow(mask_img, cmap=cm)
    plt.colorbar(im)
    plt.clim(0, 29000)
    plt.axis('off')
    plt.show()
    # plt.close()
    if save_name is not None:
        plt.savefig('{}.png'.format(save_name), bbox_inches='tight')

def plot_depth_test():
    print('ploting masks...')
    dataset_dir = '/Users/cainan/Desktop/active_perception/EVIMO/data/samsung_mono/sfm/eval'
    sequences = ['scene_03_00_000000', 'scene_03_01_000000', 'scene_03_02_000000', 'scene_03_02_000001',
                'scene_03_02_000002', 'scene_03_02_000003', 'scene_03_03_000000', 'scene_03_03_000001',
                'scene_03_03_000002', 'scene_03_04_000000']
    sequence = sequences[0]    
    mask_frame_id = 673
    save_name = None

    depth_data = np.load(os.path.join(dataset_dir + '/' + sequence, 'dataset_depth.npz'))
    depth_img = get_frame_by_index(depth_data,mask_frame_id)

    # using plt method to visualise  
    # plt.imshow(depth_img, cmap='viridis')  # 'viridis' colormap for better visualization
    plt.imshow(depth_img, cmap='gray')  # Use 'gray' colormap for grayscale visualization

    plt.colorbar()  # Display a colorbar for the depth scale
    plt.axis('off')
    plt.show()

    # # using cv2 method to visualise
    # MAX_VIS_DEPTH = 1.5
    # depth_img = np.clip(depth_img.astype(np.float32) * (255 / MAX_VIS_DEPTH / 1000), 0.0, 255.0).astype(np.uint8)
    # depth_bgr = cv2.cvtColor(depth_img, cv2.COLOR_GRAY2BGR)
    # cv2.imshow('depth img',depth_bgr)
    # k = cv2.waitKey(0)
    # if k == 27:         # ESC
    #     cv2.destroyAllWindows() 
    # save_name = None  
    # if save_name is not None:
    #     cv2.imwrite(os.path.join(save_name,str(mask_frame_id)+'depth.png'),depth_bgr)

def mask_and_depth_test():
    dataset_dir = '/Users/cainan/Desktop/active_perception/EVIMO/data/samsung_mono/sfm/eval'
    sequences = ['scene_03_00_000000', 'scene_03_01_000000', 'scene_03_02_000000', 'scene_03_02_000001',
                'scene_03_02_000002', 'scene_03_02_000003', 'scene_03_03_000000', 'scene_03_03_000001',
                'scene_03_03_000002', 'scene_03_04_000000']
    sequence = sequences[0]    
    gt_frame_id = 673
    masks = np.load(os.path.join(dataset_dir + '/' + sequence, 'dataset_mask.npz'))
    depth = np.load(os.path.join(dataset_dir + '/' + sequence, 'dataset_depth.npz'))
    m = get_frame_by_index(masks,gt_frame_id)
    d = get_frame_by_index(depth,gt_frame_id)

    # Create a figure and two subplots side-by-side
    plt.figure(figsize=(20, 5))

    # Plot mask image in the first subplot
    plt.subplot(1, 2, 1)
    # plt.title('Mask Image')
    plt.title(f'Mask Image (Frame ID: {gt_frame_id})') 
    colors = ['xkcd:light grey',
            'xkcd:salmon',
            'xkcd:plum',
            'xkcd:lightblue',
            'xkcd:gold',
            'xkcd:darkblue',
            'xkcd:white']
    cm = LinearSegmentedColormap.from_list('foo', colors[0:29], 29)
    im_mask = plt.imshow(m, cmap=cm)
    plt.clim(0, 29000)
    plt.axis('off')
    plt.colorbar(im_mask)

    # Plot depth image in the second subplot
    plt.subplot(1, 2, 2)
    plt.title('Depth Image')
    im_depth = plt.imshow(d, cmap='gray')
    plt.axis('off')
    plt.colorbar(im_depth)

    plt.show()

def plot_mask_depth(gt_frame_id,m,d):
    # Create a figure and two subplots side-by-side
    plt.figure(figsize=(20, 5))

    # Plot mask image in the first subplot
    plt.subplot(1, 2, 1)
    # plt.title('Mask Image')
    plt.title(f'Mask Image (Frame ID: {gt_frame_id})') 
    colors = ['xkcd:light grey',
            'xkcd:salmon',
            'xkcd:plum',
            'xkcd:lightblue',
            'xkcd:gold',
            'xkcd:darkblue',
            'xkcd:white']
    cm = LinearSegmentedColormap.from_list('foo', colors[0:29], 29)
    im_mask = plt.imshow(m, cmap=cm)
    plt.clim(0, 29000)
    plt.axis('off')
    plt.colorbar(im_mask)

    # Plot depth image in the second subplot
    plt.subplot(1, 2, 2)
    plt.title('Depth Image')
    # im_depth = plt.imshow(d, cmap='gray')
    im_depth = plt.imshow(d, cmap='viridis', vmin=200, vmax=900)
    # im_depth = plt.imshow(d, cmap='viridis')
    plt.axis('off')
    plt.colorbar(im_depth)
    plt.show()

def show_bunch_of_gt():
    dataset_dir = '/Users/cainan/Desktop/active_perception/EVIMO/data/samsung_mono/sfm/eval'
    sequences = ['scene_03_00_000000', 'scene_03_01_000000', 'scene_03_02_000000', 'scene_03_02_000001',
                'scene_03_02_000002', 'scene_03_02_000003', 'scene_03_03_000000', 'scene_03_03_000001',
                'scene_03_03_000002', 'scene_03_04_000000']
    sequence = sequences[0]    
    # gt_frame_id = q
    masks = np.load(os.path.join(dataset_dir + '/' + sequence, 'dataset_mask.npz'))
    depth = np.load(os.path.join(dataset_dir + '/' + sequence, 'dataset_depth.npz'))
    for gt_frame_id in range(300,701,1):  # 390 good  560 610
        m = get_frame_by_index(masks,gt_frame_id)
        d = get_frame_by_index(depth,gt_frame_id)
        # plot_mask_depth(gt_frame_id,m,d)

def plot_avg_depth_ts():
    dataset_path = '/Users/cainan/Desktop/active_perception/EVIMO/data/samsung_mono/sfm/eval'
    sequence = 'scene_03_00_000000'
    object_id = 5000
    path_depth = '{}/{}/dataset_depth.npz'.format(dataset_path, sequence)
    depth = np.load(path_depth, allow_pickle=True)
    path_mask = '{}/{}/dataset_mask.npz'.format(dataset_path, sequence)
    mask = np.load(path_mask)
    path_info = '{}/{}/dataset_info.npz'.format(dataset_path, sequence)
    meta = np.load(os.path.join(path_info), allow_pickle=True)['meta'].item()

    # get timestamps of gt frames
    timestamps = []
    frame_infos = meta['frames']
    for frame in frame_infos:
        if 'gt_frame' in frame:
            timestamps.append(frame['ts'])
    timestamps = np.array(timestamps)

    # get depth of specified object
    # avg_obj is nan if specified object is not present
    avg_obj_depth = []
    for t in timestamps:
        if not (t < timestamps[0] or t > timestamps[-1] + 1 / 60.0):
            i = np.searchsorted(timestamps, t, side='right') - 1
            d = get_frame_by_index(depth, i)
            d = d/1000 #convert depth from mm into m
            m = get_frame_by_index(mask, i)
            m = m == object_id
            masked_depth = m*d
            avg_obj_depth.append(sum(map(sum, masked_depth)) / sum(map(sum, m)))

    plt.plot(timestamps, avg_obj_depth, label='Object Depth')
    plt.legend()
    plt.show()


def viewer_est_relative_depth_img_npy():
    # sequences = ['scene_03_00_000000', 'scene_03_01_000000', 'scene_03_02_000000', 'scene_03_02_000001',
    #              'scene_03_02_000002', 'scene_03_02_000003', 'scene_03_03_000000', 'scene_03_03_000001',
    #              'scene_03_03_000002', 'scene_03_04_000000']
    sequences = ['scene17_d_01_000000', 'static_objects_dark_00_000003', 'static_objects_dark_00_000002', 
                'static_objects_dark_00_000000', 'static_objects_dark_00_000001', 'scene17_d_02_000000']

    # result_path = '/Users/cainan/Desktop/active_perception/event_depth_esti/code/stppp_and_output/x-student-events/output/2024-01-12stppp_evimo_rel'
    # result_path = '/Users/cainan/Desktop/active_perception/event_depth_esti/code/stppp_and_output/x-student-events/output/2024-01-26_newmethod_0.05dt'
    # result_path = '/Users/cainan/Desktop/active_perception/event_depth_esti/code/emvs/emvs_output/est_average_depth_img/2024-02-01_emvs_evimo_average_ll2s'
    result_path = '/Users/cainan/Desktop/active_perception/event_depth_esti/code/e2depth/e2depth_output/2024-02-07_e2depth_evimo_average_rela'
    # result_path = '/Users/cainan/Desktop/active_perception/event_depth_esti/code/e2depth/e2depth_output/2024-01-29_e2depth_evimo_average'
    sequence = sequences[5]
    imgs_path = glob.glob(os.path.join(result_path,"*"+sequence + "*" , '*.npy'))   # 'evimo_scene_03_02_000001allev_0_2'    +'est' + '*'
    if len(imgs_path) < 1:
        print('!!!!len(imgs_path) < 1!!!!!')
        print(imgs_path)
        quit()
    # quit()
    imgs_path.sort(key=lambda x: float(x.split('/')[-1][:-4]))
    for i in range(len(imgs_path)-1):
        # i = i +1 
        print(imgs_path[i])
        sys.stdout.flush()
        # continue
        est_relative_depth_img = np.load(imgs_path[i])
        plt.figure(figsize=(10, 6))
        cm_depth ='coolwarm' #'viridis'  'jet' 'coolwarm'
        min_value = np.min(est_relative_depth_img) 
        max_value = np.max(est_relative_depth_img)
        print("Minimum Value:", min_value, "Maximum Value:", max_value)

        plt.imshow(est_relative_depth_img, cmap=cm_depth) # , vmin=min_value, vmax=max_value)
        temp = imgs_path[i].split('/')[-1][:-4]
        plt.title(f'est_relative_depth_img frame {temp}') 
        plt.colorbar()  # Display a colorbar for the depth scale
        plt.axis('off')
        plt.show()  

# def plot_recover_e2depth():
    

if __name__ == '__main__':    
    # plot_mask_test() 
    # plot_depth_test()
    # mask_and_depth_test()
    # show_bunch_of_gt()
    viewer_est_relative_depth_img_npy() 
