import numpy as np
from matplotlib import pyplot as plt
from rela_utils.visual_util import *

def get_object_ids(mask):
    object_ids = np.unique(mask)
    # check number of pixels of an object
    # for o in list(object_ids):
    #     mask_obj = mask == o
        # if np.sum(mask_obj) <= 50: # TODO ask pia about this
        #     object_ids = np.delete(object_ids, np.where(object_ids == o))

    # remove background and table
    object_ids = np.delete(object_ids, np.where(object_ids == 0))
    object_ids = np.delete(object_ids, np.where(object_ids == 22000))
    return object_ids


def compute_avg_obj_depth_dict(m, d, object_ids,show = 0): 
    avg_obj_depth_dict = {}
    for object_id in object_ids:
        selected_pixels = d[(m == object_id)&(d!=0)]
        if len(selected_pixels) > 0:
            avg_obj_depth = np.mean(selected_pixels) 
        else:
            print('!!!!!!!len(selected_pixels) <= 0!!!!!!')
            # quit()  # see this condition have or not, TODO delete this condition maybe
            continue
        avg_obj_depth_dict[object_id] = avg_obj_depth

        if show:
            # Plot the heatmap of average depth values for the specified object
            cm = 'viridis'
            avg_image = np.where(m == object_id,avg_obj_depth,0)
            im_depth = plt.imshow(avg_image, cmap=cm)
            plt.title(f'Object ID: {object_id}\nAverage Depth: {avg_obj_depth:.2f}')
            plt.axis('off')
            plt.colorbar(im_depth)
            plt.show()
            
    if all(value == 0 for value in avg_obj_depth_dict.values()):
        print('all(value == 0 for value in avg_obj_depth_dict.values())')

        # plot_mask_depth(0,m,0,d,plot_FLAG = 0)
        
        # est_average_depth_img = np.zeros_like(m)
        # for object_id in avg_obj_depth_dict:
        #     est_average_depth_img = np.where(m == object_id,avg_obj_depth_dict[object_id],est_average_depth_img)

        # e2depth_plot(est_average_depth_img,0,avg_obj_depth_dict,show = 0) 

    return avg_obj_depth_dict
    
def compute_avg_obj_depth_dict_honey(m, d, object_ids,show = 0): 
    avg_obj_depth_dict = {}
    for object_id in object_ids:
        selected_pixels = d[(m == object_id)&(d!=0)]
        if len(selected_pixels) > 0:
            avg_obj_depth = np.mean(selected_pixels) 
        else:
            print('!!!!!!!len(selected_pixels) <= 0!!!!!!')
            # quit()  # see this condition have or not, TODO delete this condition maybe
            continue
        avg_obj_depth_dict[object_id] = avg_obj_depth

        if show:
            # Plot the heatmap of average depth values for the specified object
            cm = 'viridis'
            avg_image = np.where(m == object_id,avg_obj_depth,0)
            im_depth = plt.imshow(avg_image, cmap=cm)
            plt.title(f'Object ID: {object_id}\nAverage Depth: {avg_obj_depth:.2f}')
            plt.axis('off')
            plt.colorbar(im_depth)
            plt.show()
            
    if all(value == 0 for value in avg_obj_depth_dict.values()):
        print('all(value == 0 for value in avg_obj_depth_dict.values())')

    return avg_obj_depth_dict


def compute_avg_obj_depth_dict_EMVS(m, d, object_ids,show = 0):  # TODO maybe need to remove the background when e2depth
    avg_obj_depth_dict = {}
    background = np.max(d)
    for object_id in object_ids:
        # selected_pixels = d[(m == object_id)]
        selected_pixels = d[(m == object_id) & (d != background)]
        if len(selected_pixels) > 0:
            avg_obj_depth = np.mean(selected_pixels) 
        else:
            print('!!!!!!!len(selected_pixels) <= 0!!!!!!')
            # quit()  # see this condition have or not, TODO delete this condition maybe
            continue
        avg_obj_depth_dict[object_id] = avg_obj_depth
        if show:
            # Plot the heatmap of average depth values for the specified object
            cm = 'viridis'
            avg_image = np.where(m == object_id,avg_obj_depth,0)
            im_depth = plt.imshow(avg_image, cmap=cm)
            plt.title(f'Object ID: {object_id}\nAverage Depth: {avg_obj_depth:.2f}')
            plt.axis('off')
            plt.colorbar(im_depth)
            plt.show()
            
    if all(value == 0 for value in avg_obj_depth_dict.values()):
        print('all(value == 0 for value in avg_obj_depth_dict.values())')        
        est_average_depth_img = np.zeros_like(m)
        for object_id in avg_obj_depth_dict:
            est_average_depth_img = np.where(m == object_id,avg_obj_depth_dict[object_id],est_average_depth_img)

    return avg_obj_depth_dict

def parse_gt_data(dataset_dir, sequence): # 3 npz from evimo data
    # load meta data
    meta = np.load(os.path.join(dataset_dir, sequence, 'dataset_info.npz'), allow_pickle=True)['meta'].item()
    # load masks timestamps
    frame_infos = meta['frames']
    masks_timestamps = []
    events_t = np.load(os.path.join(dataset_dir, sequence, 'dataset_events_t.npy'), mmap_mode='r')
    t_ref_events = events_t[0]
    print('t_ref_events',t_ref_events)
    del events_t
    for frame in frame_infos:
        if 'gt_frame' in frame:
            masks_timestamps.append(frame['ts']-t_ref_events)

    mask_npz = np.load(os.path.join(dataset_dir, sequence, 'dataset_mask.npz'))
    mask_names = list(mask_npz.keys())
    mask_names.sort()
    
    gt_depth_npz = np.load(os.path.join(dataset_dir, sequence, 'dataset_depth.npz'))
    gt_depth_names = list(gt_depth_npz.keys())
    gt_depth_names.sort()

    return masks_timestamps,mask_npz,mask_names,gt_depth_npz,gt_depth_names



def parse_evimo_masks(dataset_path,sequence):
    path_mask = '{}/{}/dataset_mask.npz'.format(dataset_path, sequence)
    path_info = '{}/{}/dataset_info.npz'.format(dataset_path, sequence)
    meta = np.load(path_info, allow_pickle=True)['meta'].item()
    frame_infos = meta['frames']
    masks_timestamps = []
    for frame in frame_infos:
        if 'gt_frame' in frame:
            masks_timestamps.append(frame['ts'])
    masks_timestamps = np.array(masks_timestamps).astype(np.float64) 
    masks = np.load(path_mask)
    masks_names = list(masks.keys())
    masks_names.sort()
    events_t = np.load(os.path.join(dataset_path, sequence, 'dataset_events_t.npy'), mmap_mode='r')
    t_ref_events = events_t[0]
    print('t_ref_events',t_ref_events)

    return masks_timestamps,masks_names,masks,t_ref_events



def compute_gt_relative_distance_dict(m, d, object_ids):
    # compute and plot the gt distance
    avg_obj_depth_dict = compute_avg_obj_depth_dict(m, d, object_ids,show = 0)
    # print('avg_obj_depth_dict',avg_obj_depth_dict)
    # sortedDict = sorted(avg_obj_depth_dict.items(), key=lambda x: x[1])
    # print('sortedDict',sortedDict)

    # choose largest object as reference object
    size_current = 0
    reference_object_id = 0
    for obj_id in avg_obj_depth_dict.keys():
        mask_obj = m == obj_id
        if np.sum(mask_obj) > size_current:
            size_current = np.sum(mask_obj)
            reference_object_id = obj_id
    if reference_object_id == 0:
        return None,None


            
    # reference_object_id = max(avg_obj_depth_dict.keys(), key=lambda obj_id: np.sum(m == obj_id))
    # if not avg_obj_depth_dict:  # Check if avg_obj_depth_dict is empty
    #     print("Warning: No objects found. Skipping this frame.")
    #     continue  # Skip the rest of the loop for this iteration
    # compute !relative! 
    gt_relative_distance_dict = {key: value / avg_obj_depth_dict[reference_object_id] for key, value in avg_obj_depth_dict.items()}
    return gt_relative_distance_dict, reference_object_id




def find_corresponding_mask(masks_timestamps,masks_names,masks,depth_frame_id,depth_ts):   # masks_timestamps,masks_names,masks
    mask_frame_id = np.argmin(np.abs(masks_timestamps - depth_ts))
    mask_timestamp = masks_timestamps[mask_frame_id]
    mask_name = masks_names[mask_frame_id]
    m = masks[mask_name]
    # print(f'mask_frame_id {mask_frame_id}, depth map id {depth_frame_id}, nearest mask timestamp {mask_timestamp}, depth map ts {depth_ts}')
    return m,mask_frame_id,mask_timestamp,mask_frame_id





