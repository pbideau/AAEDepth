from Estimators.EstimatorFactory import EstimatorFactory
from Estimators.Estimator import Estimator
import os
from utils.utils import *
from utils.load import load_dataset
import argparse
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import torch
import numpy as np
import pandas as pd
from copy import deepcopy
import cProfile


def events2frame(warped_events_batch, height, width, padding=0):

    min_x = -padding
    min_y = -padding
    new_height = height + 2 * padding
    new_width = width + 2 * padding

    pos_batch = warped_events_batch[warped_events_batch[:, 3] == 1]
    neg_batch = warped_events_batch[warped_events_batch[:, 3] == 0]
    frame1 = accumulate_event(pos_batch, min_x=min_x, min_y=min_y,
                                   height=new_height, width=new_width)
    frame0 = accumulate_event(neg_batch, min_x=min_x, min_y=min_y,
                                   height=new_height, width=new_width)

    frame = np.stack((frame1, frame0))
    return frame

def accumulate_event(events, **kwargs):
    min_x = kwargs['min_x']
    min_y = kwargs['min_y']
    height = kwargs['height']
    width = kwargs['width']

    events[:, 1] = events[:, 1] - min_x
    events[:, 2] = events[:, 2] - min_y
    xx = np.floor(events[:, 1])
    yy = np.floor(events[:, 2])
    dx = events[:, 1] - xx
    dy = events[:, 2] - yy

    H = height + 1
    W = width + 1

    zero_v = 0
    ones_v = 1
    index = np.where(xx >= 0, ones_v, zero_v) * np.where(yy >= 0, ones_v, zero_v) * np.where(
        xx < (W - 1), ones_v, zero_v) * np.where(yy < (H - 1), ones_v, zero_v)
    index = index.astype(bool)
    xx = xx[index].astype(int)
    yy = yy[index].astype(int)
    dx = dx[index]
    dy = dy[index]
    events[events[:, 3] == 0, 3] = -1
    polarity = events[index, 3]
    iwe = accumulate_operation(xx, yy, dx, dy, polarity, H, W)
    return iwe[:-1, :-1]

def accumulate_operation(pxx, pyy, dx, dy, polarity, height, width):
    iwe = np.zeros((height, width))

    np.add.at(iwe, (pyy, pxx), polarity * (1. - dx) * (1. - dy))
    np.add.at(iwe, (pyy, pxx + 1), polarity * dx * (1. - dy))
    np.add.at(iwe, (pyy + 1, pxx), polarity * (1. - dx) * dy)
    np.add.at(iwe, (pyy + 1, pxx + 1), polarity * dx * dy)
    return iwe


def load_events(folder):
    events_t  = np.load(os.path.join(folder, 'dataset_events_t.npy'), mmap_mode='r')
    events_xy = np.load(os.path.join(folder, 'dataset_events_xy.npy'), mmap_mode='r')
    events_p  = np.load(os.path.join(folder, 'dataset_events_p.npy'), mmap_mode='r')

    t_ref_events = events_t[0]
    events_t = events_t - t_ref_events

    events_t = np.atleast_2d(events_t.astype(np.float32)).transpose()
    events_p = np.atleast_2d(events_p.astype(np.float32)).transpose()

    events = np.hstack((events_t,
                        events_xy.astype(np.float32),
                        events_p))
    return events, t_ref_events

def get_object_ids(mask):
    object_id = []

    # get all object ids
    try: #evimo
        for i in mask.files:
            objects = np.unique(mask[i])
            object_id = np.append(object_id, objects)
            object_id = np.unique(object_id)
    except: #scioi_events
        for mask_id, mask_info in mask.items():
            objects = np.unique(mask[mask_id])
            object_id = np.append(object_id, objects)
            object_id = np.unique(object_id)

    # remove background and table
    object_id = np.delete(object_id, np.where(object_id == 0))
    object_id = np.delete(object_id, np.where(object_id == 22000))

    return object_id.astype(int)

def plot_masks(mask_img, save_name):
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
    # plt.show()
    plt.savefig('{}.png'.format(save_name), bbox_inches='tight')

def get_frame_by_index(frames, index):
    frames_names = list(frames.keys())
    frames_names.sort()
    frame_name = frames_names[index]
    return np.copy(frames[frame_name]) # To extract and keep in RAM

def filter_events(events, obj_id, masks, timestamps, width, height):
    masks_names = list(masks.keys())
    masks_names.sort()
    idx = 0
    masked = []
    for i in range(0, len(timestamps)):
        mask_name = masks_names[i]
        mask = masks[mask_name]
        mask_obj = mask == obj_id

        if i == len(timestamps)-1:
            idx_end = len(events)
        else:
            t_end = 0.5 * (timestamps[i] + timestamps[i + 1])
            idx_end = np.argmin(np.abs(events[:, 0] - t_end))

        #m = [True if mask_obj[int(y), int(x)] else False for (_, x, y, _) in events[idx: idx_end]]

        m = []
        for (_, x, y, _) in events[idx: idx_end]:
            y = int(min(y, height-1))
            y = int(max(y, 0))
            x = int(min(x, width-1))
            x = int(max(x, 0))
            if mask_obj[int(y), int(x)]:
                m.append(True)
            else:
                m.append(False)

        masked.extend(m)
        idx = idx_end

    return masked


if __name__ == '__main__':
    profiler = cProfile.Profile()
    profiler.enable()
    parser = argparse.ArgumentParser(description='Demo code to extract motion from eventdata')
    parser.add_argument('-p', '--path', default='dataset', help="Path to dataset")
    parser.add_argument('-sv', '--save', default='dataset', help="Path to saved output")
    parser.add_argument('-d', '--dataset', default='DAVIS_240C', help="Name of dataset, default as 'DAVIS_240C'")
    parser.add_argument('-s', '--seq', default=0, help="Integer indexing the sequnece array, default as 'dynamic_rotation'")
    parser.add_argument('-o', '--output', default='output', help="Name for output file, default as 'output.txt'")
    parser.add_argument('-n', '--Ne', default=30000, help="The number of events per batch, default as 30000")
    parser.add_argument('-dt', '--dtime', default=0.025, help="The time interval (in sec) for one event batch, default as 0.025sec")
    parser.add_argument('-a', '--alpha', default=0.1, help="'alpha' of gamma prior, default as 0.1")
    parser.add_argument('-b', '--beta', default=1.59, help="'beta' of gamma prior, default as 1.59")
    parser.add_argument('-l', '--lr', default=0.1, help="Learning rate of optimization, defualt as 0.05")
    parser.add_argument('-i', '--iter', default=250, help="Maximum number of iterations, default as 250")
    parser.add_argument('-m', '--method', default='st-ppp', help="The name of method, can be selected from ['st-ppp', 'cmax'], default as 'st-ppp'")
    parser.add_argument('-t', '--transformation', default='rot', help="The type of transformation, can be selected from ['rot', 'trans'], default as 'rot'")
    parser.add_argument('-f', '--figure', action='store_true', default=False, help="Save figures or not, default as False, use '-f' to set the flag")
    parser.add_argument('-lm', '--load_mask', default=False, help="load predifined segmentation masks from file")
    parser.add_argument('-am', '--apply_mask', default=False, help="use masks for alignment - either predefined or using the histogram approach to find area with most events")
    parser.add_argument('-id', '--object_id', help="object id specifing a single object in segmentation masks that possibly contain multiple segmented objects")

    args = parser.parse_args()
    dataset_path = args.path
    output_path = args.save
    dataset = args.dataset
    sequence_idx = int(args.seq)
    Ne = int(args.Ne)
    dt = float(args.dtime)
    alpha = float(args.alpha)
    beta = float(args.beta)
    lr = float(args.lr)
    iters = int(args.iter)
    method = args.method
    trans_type = args.transformation
    save_figs = args.figure
    apply_mask = args.apply_mask #eval(args.apply_mask)
    load_mask = args.load_mask #eval(args.load_mask)
    object_id = args.object_id

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    if bool(dataset == 'evimo') | bool(dataset == 'evimo_ll'):

        if dataset == 'evimo':
            sequences = ['scene_03_00_000000', 'scene_03_01_000000', 'scene_03_02_000000', 'scene_03_02_000001',
                         'scene_03_02_000002', 'scene_03_02_000003', 'scene_03_03_000000', 'scene_03_03_000001',
                         'scene_03_03_000002', 'scene_03_04_000000']
        else:
            sequences = ['scene17_d_01_000000', 'scene17_d_02_000000', 'static_objects_dark_00_000000', 
                         'static_objects_dark_00_000001', 'static_objects_dark_00_000002', 'static_objects_dark_00_000003']
        sequence = sequences[sequence_idx]

        LUT, fx, fy, px, py, width, height = load_dataset(dataset, dataset_path, sequence)
        #LUT = torch.from_numpy(LUT).type(torch.float32).to(device)
        ALIGN = Estimator(height=height, width=width, fx=fx, fy=fy, px=px, py=py)

        # load meta data
        meta = np.load(os.path.join(dataset_path + '/' + sequence, 'dataset_info.npz'), allow_pickle=True)['meta'].item()

        # load events
        events, t_ref_events = load_events(dataset_path + '/' + sequence)

        end_time = events[0][0] + 1.0
        events = events[events[:, 0] <= end_time]

        # load masks with timestamps
        mask = np.load(os.path.join(dataset_path + '/' + sequence, 'dataset_mask.npz'))
        frame_infos = meta['frames']
        timestamps_masks = []
        for frame in frame_infos:
            if 'gt_frame' in frame:
                timestamps_masks.append(frame['ts'] - t_ref_events)
        timestamps_masks = np.array(timestamps_masks)

        #'''
        # undo camera rotation (given the imu info)
        #events_proc = deepcopy(undistortion(events, LUT, events[0][0]))
        events_proc = deepcopy(events)
        events_proc = undistortion(events_proc, LUT, events[0][0])

        #t_start = 0.15 #13.5 #17.0
        #t_end = 0.2 #13.55 #17.05
        #idx_img = np.argmin(np.abs(events_proc[:, 0] - t_start))
        #idx_end_img = np.argmin(np.abs(events_proc[:, 0] - t_end))
        #frame_orig = events2frame(events_proc[idx_img: idx_end_img, :], height, width)
        #event_img_orig = frame_orig.sum(axis=0)

        events_proc = torch.as_tensor(events_proc, dtype=torch.float, device=device)

        imu_left = meta['imu']['/prophesee/left/imu']
        timestamps_imu = []
        #y = []
        for i in imu_left:
            timestamps_imu.append(i['ts'] - t_ref_events)
            #y.append(i['angular_velocity']['y'])

        count = 1
        while True:

            t_start = events_proc[0, 0] + dt * (count - 1)
            t_end = events_proc[0, 0] + dt * count

            t_imu = 0.5 * (t_start + t_end)
            t = np.argmin(np.abs(np.asarray(timestamps_imu) - t_imu.item()))
            x, y, z = imu_left[t]['angular_velocity']['x'], imu_left[t]['angular_velocity']['y'], \
                imu_left[t]['angular_velocity']['z']
            #angular_vel = torch.as_tensor([y, -x, z], dtype=torch.float, device=device)
            angular_vel = torch.as_tensor([y, -x, z], dtype=torch.float, device=device)

            idx_start = torch.argmin(torch.abs(events_proc[:, 0] - t_start))
            idx_end = torch.argmin(torch.abs(events_proc[:, 0] - t_end))
            events_batch = deepcopy(events_proc[idx_start: idx_end])

            t_ref = events_proc[idx_start, 0]
            events_batch[:, 0] = events_batch[:, 0] - t_ref

            if events_batch.size()[0] == 1:
                events_proc[idx_start: idx_end, 1:] = ALIGN.warp_event(angular_vel, events_batch)[1:]
            else:
                events_proc[idx_start: idx_end, 1:] = ALIGN.warp_event(angular_vel, events_batch)[:, 1:]

            idx = idx_end

            if t_end > events_proc[-1,0]:
                break

            print(timestamps_imu[t])
            count += 1

        events_proc = events_proc.detach().cpu().numpy()

        '''
        idx = 0
        for t in range(0, len(timestamps_imu) - 1):
            t_end = 0.5 * (timestamps_imu[t] + timestamps_imu[t + 1])
            idx_end = torch.argmin(torch.abs(events_proc[:, 0] - t_end))
            x, y, z = imu_left[t]['angular_velocity']['x'], imu_left[t]['angular_velocity']['y'], \
            imu_left[t]['angular_velocity']['z']
            angular_vel = torch.as_tensor([x, y, z], dtype=torch.float, device=device)
            if idx != idx_end: #more than no events needed
                events_batch = deepcopy(events_proc[idx: idx_end])
                t_ref = events_proc[idx, 0]
                events_batch[:, 0] = events_batch[:, 0] - t_ref

                if events_batch.size()[0] == 1:
                    events_proc[idx: idx_end, 1:] = ALIGN.warp_event(angular_vel, events_batch)[1:]
                else:
                    events_proc[idx: idx_end, 1:] = ALIGN.warp_event(angular_vel, events_batch)[:, 1:]
                idx = idx_end
                print(timestamps_imu[t])
        events_proc = events_proc.detach().cpu().numpy()
        '''

        # verify undoing rotational camera motion
        #frame = events2frame(events_proc[idx_img: idx_end_img, :], height, width)
        #event_img = frame.sum(axis=0)
        #plt.figure()
        #plt.subplot(1, 2, 1)
        #plt.title('rotated events')
        #plt.imshow(event_img, cmap='bwr')
        #plt.clim(-4, 4)
        #plt.subplot(1, 2, 2)
        #plt.title('orig. events')
        #plt.imshow(event_img_orig, cmap='bwr')
        #plt.clim(-4, 4)
        #plt.savefig('/Users/piabideau/Desktop/test/rotCom.png')

    elif dataset == 'DAVIS_240C':

        sequences = ['boxes_rotation', 'dynamic_rotation', 'poster_rotation', 'shapes_rotation',
                     'boxes_translation', 'dynamic_translation', 'poster_translation', 'shapes_translation']
        # sequences = ['dice_01-1', 'dice_01-2', 'dice_02-1', 'dice_02-2', 'dice_03-1', 'dice_03-2', 'dice_04-1', 'dice_04-2',
        #             'dice_05-1', 'dice_05-2', 'dice_06-1', 'dice_06-2', 'dice_07-1', 'dice_07-2', 'dice_08-1', 'dice_08-2',
        #             'dice_09-1', 'dice_09-2']
        sequence = sequences[sequence_idx]

        LUT, fx, fy, px, py, width, height = load_dataset(dataset, dataset_path, sequence)

        events = pd.read_csv(
            '{}/{}/events.txt'.format(dataset_path, sequence), sep=" ", header=None)
        events.columns = ["ts", "x", "y", "p"]
        events = events.to_numpy()

        events_proc = deepcopy(events)
        events_proc = undistortion(events_proc, LUT, events[0][0])

    elif dataset == 'scioi_event':

        sequences = ['trans_x_slow', 'trans_x_fast', 'trans_xy_slow', 'trans_xy_fast', 'freeMotion']
        sequence = sequences[sequence_idx]

        LUT, fx, fy, px, py, width, height = load_dataset(dataset, dataset_path, sequence)
        # LUT = torch.from_numpy(LUT).type(torch.float32).to(device)
        ALIGN = Estimator(height=height, width=width, fx=fx, fy=fy, px=px, py=py)

        # load meta data
        meta = np.load(os.path.join(dataset_path + '/' + sequence, 'dataset_info.npz'), allow_pickle=True)[
            'meta'].item()

        # load events
        events, t_ref_events = load_events(dataset_path + '/' + sequence)
        # '''
        # undo camera rotation (given the imu info)
        # events_proc = deepcopy(undistortion(events, LUT, events[0][0]))
        events = undistortion(events, LUT, events[0][0])
        events_proc = deepcopy(events)

        t_start = 4.0 #7.5 #4.0 #scene3 #1.2 #11.0 #scene1 #4.6 #18.0 #8.5
        t_end = 4.025 #7.525 #4.025 #scene3 #1.25 #11.05 #scene1 #4.65 #18.05 #8.55
        idx_img = np.argmin(np.abs(events_proc[:, 0] - t_start))
        idx_end_img = np.argmin(np.abs(events_proc[:, 0] - t_end))
        frame_orig = events2frame(events_proc[idx_img: idx_end_img, :], height, width)
        event_img_orig = frame_orig.sum(axis=0)

        events_proc = torch.as_tensor(events_proc, dtype=torch.float, device=device)

        # load masks with timestamps
        mask = np.load(os.path.join(dataset_path + '/' + sequence, 'dataset_mask.npz'), allow_pickle=True)['mask_id'].item()
        timestamps_masks = np.load(os.path.join(dataset_path + '/' + sequence, 'dataset_mask.npz'), allow_pickle=True)['timestamp']
        timestamps_masks = timestamps_masks - t_ref_events

        #for mask_id, mask_info in mask.items():
        #    objects = mask[mask_id]

        # load imu data
        imu = np.load(os.path.join(dataset_path + '/' + sequence, 'dataset_info.npz'), allow_pickle=True)['imu'].item()
        timestamps_imu = []
        x_vel = []
        y_vel = []
        z_vel = []
        for imu_id, imu_info in imu.items():
            timestamps_imu.append(imu_info['ts'] - t_ref_events)
            x, y, z = imu_info['angular velocity']['x'], imu_info['angular velocity']['y'], imu_info['angular velocity']['z']
            x_vel.append(x)
            y_vel.append(y)
            z_vel.append(z)

        count = 1
        while True:

            t_start = events_proc[0, 0] + dt * (count - 1)
            t_end = events_proc[0, 0] + dt * count
            t_imu = 0.5 * (t_start + t_end)
            t = np.argmin(np.abs(np.asarray(timestamps_imu) - t_imu.item()))
            angular_vel = torch.as_tensor([x_vel[t], y_vel[t], z_vel[t]], dtype=torch.float, device=device)
            #angular_vel = torch.as_tensor([0, y_vel[t], 0], dtype=torch.float, device=device)

            idx_start = torch.argmin(torch.abs(events_proc[:, 0] - t_start))
            idx_end = torch.argmin(torch.abs(events_proc[:, 0] - t_end))
            events_batch = deepcopy(events_proc[idx_start: idx_end])

            t_ref = events_proc[idx_start, 0]
            events_batch[:, 0] = events_batch[:, 0] - t_ref

            if events_batch.size()[0] == 1:
                events_proc[idx_start: idx_end, 1:] = ALIGN.warp_event(angular_vel, events_batch)[1:]
            else:
                events_proc[idx_start: idx_end, 1:] = ALIGN.warp_event(angular_vel, events_batch)[:, 1:]

            idx = idx_end

            if t_end > events_proc[-1, 0]:
                break

            print(timestamps_imu[t])
            count += 1

        events_proc = events_proc.detach().cpu().numpy()

        '''
        idx = 0
        t = 0
        #for t in range(0, len(timestamps_imu) - 1):
        for imu_id, imu_info in imu.items():
            if t == len(timestamps_imu)-1:
                idx_end = len(events_proc)
            else:
                t_end = 0.5 * (timestamps_imu[t] + timestamps_imu[t + 1])
                idx_end = torch.argmin(torch.abs(events_proc[:, 0] - t_end))
            x, y, z = imu_info['angular velocity']['x'], imu_info['angular velocity']['y'], \
                imu_info['angular velocity']['z']
            angular_vel = torch.as_tensor([x, y, z], dtype=torch.float, device=device)
            events_batch = deepcopy(events_proc[idx: idx_end])
            t_ref = events_proc[idx, 0]
            events_batch[:, 0] = events_batch[:, 0] - t_ref
            events_proc[idx: idx_end, 1:] = ALIGN.warp_event(angular_vel, events_batch)[:, 1:]
            idx = idx_end
            t = t+1
        events_proc = events_proc.detach().cpu().numpy()
        '''

        # verify undoing rotational camera motion
        frame = events2frame(events_proc[idx_img: idx_end_img, :], height, width)
        event_img = frame.sum(axis=0)
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.title('rotated events')
        plt.imshow(event_img, cmap='bwr')
        plt.clim(-4, 4)
        plt.subplot(1, 2, 2)
        plt.title('orig. events')
        plt.imshow(event_img_orig, cmap='bwr')
        plt.clim(-4, 4)
        plt.savefig('/Users/piabideau/Desktop/test/rotCom.png')

    print("Events total count: ", len(events))
    print("Time duration of the sequence: {} s".format(events[-1][0] - events[0][0]))

    #path to save results
    res_save_dir = os.path.join(output_path, sequence)
    if not os.path.exists(res_save_dir):
        os.makedirs(res_save_dir)

    VE = EstimatorFactory(fx=fx,
                          fy=fy,
                          px=px,
                          py=py,
                          method = method,
                          transformation = trans_type,
                          dataset_path=dataset_path,
                          sequence=sequence,
                          Ne=Ne,
                          height=height,  #180,  # TODO: set automatic update
                          width=width,  #240,
                          dt = dt,
                          overlap=0,
                          padding=100,
                          lr=lr,
                          iters=iters,
                          sigma_prior = 1,
                          version = 0,
                          gamma_param = [alpha, beta]
                          ).get_estimator()

    #get object id list
    if load_mask:

        # if object_id is not None:
        #     object_id = [int(object_id)]
        # else:
        #     object_id = get_object_ids(mask)

        output_filename = args.output + '_' + 'global.txt'
        save_filepath_global = os.path.join(res_save_dir, output_filename)

        if object_id is None:
            # align all objects simultaneously
            print('-----global')
            obj_mask = [True for i in range(len(events))]
            VE(events_proc, save_filepath_global, obj_mask, count=1, save_figs=save_figs)


        else:
            object_id = [int(object_id)]

            # iterate over object id list
            for obj_id in object_id:

                output_filename = args.output + '_' + f"{obj_id:08d}" + '.txt'
                save_filepath = os.path.join(res_save_dir, output_filename)

                # get events belonging to a specific object_id
                ##############################################################
                print('filter_events')
                obj_mask = filter_events(events, obj_id, mask, timestamps_masks, width, height)
                print('end filter_events')


                VE.reset(fx, fy, px, py, dataset_path, sequence, Ne, height, width, dt, 0, 100, lr, iters, 1, 0, [alpha, beta])
                print('save_filepath_global',save_filepath_global)
                event_seq = VE(events_proc, save_filepath, obj_mask, count=1, save_figs=save_figs, rot_direction_filepath=save_filepath_global)
                #events = events.detach().cpu().numpy()
                del event_seq, obj_mask
                torch.cuda.empty_cache()



    else:
        output_filename = args.output + '.txt'
        save_filepath = os.path.join(res_save_dir, output_filename)

        #events = torch.from_numpy(events).type(torch.float32).to(device)
        obj_mask = [True for i in range(len(events))]

        VE(events_proc, save_filepath, obj_mask, count=1, save_figs=save_figs)


    if load_mask and object_id:
        profiler.disable()
        profiler.dump_stats("obj_example.prof")
    elif load_mask:
        profiler.disable()
        profiler.dump_stats("example.prof")


        

