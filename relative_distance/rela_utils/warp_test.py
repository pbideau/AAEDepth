# import numpy as np
import os
from copy import deepcopy

from Estimators.Estimator import Estimator
from main import load_events   #,filter_events
from utils.utils import *
from utils.load import load_dataset

from relative_distance.relative_dis import *
from relative_distance.rela_utils.warp_util import *
from rela_utils.re_utils import *

def filter_events2(events, obj_id, masks, timestamps, width, height):
    masks_names = sorted(masks.keys())
    masked = np.array([], dtype=bool)
    idx = 0
    for i, mask_name in enumerate(masks_names):
        mask = masks[mask_name] == obj_id

        if i == len(timestamps) - 1:
            idx_end = len(events)
        else:
            t_end = 0.5 * (timestamps[i] + timestamps[i + 1])
            idx_end = np.argmin(np.abs(events[:, 0] - t_end))

        event_subset = events[idx:idx_end, 1:3].astype(int)
        event_subset[:, 0] = np.clip(event_subset[:, 0], 0, width - 1)
        event_subset[:, 1] = np.clip(event_subset[:, 1], 0, height - 1)

        m = mask[event_subset[:, 1], event_subset[:, 0]]
        masked = np.concatenate((masked, m))

        idx = idx_end

    return masked

def accumulate_event(events, fixed_size = False, **kwargs):
    device = events.device
    min_x = kwargs['min_x'].item()
    min_y = kwargs['min_y'].item()
    height = kwargs['height'].item()
    width = kwargs['width'].item()

    events[:, 1] = events[:, 1] - min_x
    events[:, 2] = events[:, 2] - min_y
    xx = events[:, 1].floor()
    yy = events[:, 2].floor()
    dx = events[:, 1] - xx
    dy = events[:, 2] - yy

    H = height + 1
    W = width + 1
    if not fixed_size:
        xx = xx.long()
        yy = yy.long()
        events[events[:, 3] == 0, 3] = -1
        polarity = events[:, 3]
        iwe = accumulate_operation(xx, yy, dx, dy, polarity, H, W, device)
        return iwe
    
    else:
        zero_v = torch.tensor([0.], device=device)
        ones_v = torch.tensor([1.], device=device)
        index = torch.where(xx>=0, ones_v, zero_v)*torch.where(yy>=0, ones_v, zero_v)*torch.where(xx<(W - 1), ones_v, zero_v)*torch.where(yy<(H - 1), ones_v, zero_v)
        index = index.bool()
        xx = xx[index].long()
        yy = yy[index].long()
        dx = dx[index]
        dy = dy[index]
        events[events[:, 3] == 0, 3] = -1
        polarity = events[index, 3]
        iwe = accumulate_operation(xx, yy, dx, dy, polarity, H, W, device)
        return iwe[:-1, :-1]

def accumulate_operation(pxx, pyy, dx, dy, polarity, height, width, device):
    iwe = torch.zeros(size = torch.Size([height, width]), device = device)
    iwe.index_put_((pyy    , pxx    ), polarity*(1.-dx)*(1.-dy), accumulate=True)
    iwe.index_put_((pyy    , pxx + 1), polarity*dx*(1.-dy), accumulate=True)
    iwe.index_put_((pyy + 1, pxx    ), polarity*(1.-dx)*dy, accumulate=True)
    iwe.index_put_((pyy + 1, pxx + 1), polarity*dx*dy, accumulate=True)
    return iwe

def events2frame(warped_events_batch, fixed_size = False, padding = 0):
    device = warped_events_batch.device
    #pos_batch = warped_events_batch[warped_events_batch[:,3]==1]
    #neg_batch = warped_events_batch[warped_events_batch[:,3]==0]
    if not fixed_size:
        min_x = torch.min(warped_events_batch[:, 1]).floor().long()
        max_x = torch.max(warped_events_batch[:, 1]).ceil().long()
        min_y = torch.min(warped_events_batch[:, 2]).floor().long()
        max_y = torch.max(warped_events_batch[:, 2]).ceil().long()
        new_height = torch.max(max_y-min_y+1, height-min_y)
        new_width = torch.max(max_x-min_x+1, width-min_x)
        
    else:
        min_x = torch.tensor(-padding, device = device)
        min_y = torch.tensor(-padding, device = device)
        new_height = torch.tensor(height + 2 * padding)
        new_width = torch.tensor(width + 2 * padding)

    if len(warped_events_batch.size())<=1:
        frame0 = torch.zeros(new_height, new_width, device=device)
        frame1 = torch.zeros(new_height, new_width, device=device)
    else:
        pos_batch = warped_events_batch[warped_events_batch[:,3]==1]
        neg_batch = warped_events_batch[warped_events_batch[:,3]==0]
        frame1 = accumulate_event(pos_batch, fixed_size=fixed_size, min_x = min_x, min_y = min_y, height=new_height, width=new_width)
        frame0 = accumulate_event(neg_batch, fixed_size=fixed_size, min_x = min_x, min_y = min_y, height=new_height, width=new_width)

    frame = torch.stack((frame1, frame0))
    return frame

if __name__ == '__main__':
    # print('1')
    est_dir_our_base = '/Users/cainan/Downloads/t_050_2dof_050'
    sequence = 'scene_03_02_000000'
    dataset_dir = '/Users/cainan/Desktop/active_perception/semester_poster/EVIMO/data/evimo_npz/samsung_mono/sfm/eval'
    
    est_dir_our = os.path.join(est_dir_our_base, sequence)
    timestamps_masks,mask_npz,mask_names,_,_ = parse_gt_data(dataset_dir, sequence)
    # print('timestamps_masks[len(timestamps_masks)-1]',timestamps_masks[len(timestamps_masks)-1])
    events, t_ref_events = load_events(dataset_dir + '/' + sequence)    
    # events = events[::15]
    dataset = 'evimo'
    LUT, fx, fy, px, py, width, height = load_dataset(dataset, dataset_dir, sequence)
    events_proc = deepcopy(events)
    events_proc = undistortion(events_proc, LUT, events[0][0])
    object_id = 5000
    # obj_mask = filter_events(events, obj_id, mask_npz, timestamps_masks, width, height)
    print('filtering')
    # print('timestamps_masks[len(timestamps_masks)-1]',timestamps_masks[len(timestamps_masks)-1])

    obj_mask = filter_events2(events, object_id, mask_npz, timestamps_masks, width, height)
    print('filter finished')

    txt_dict = get_txt_dict(est_dir_our)
    txt_file = txt_dict[object_id]

    est_freq_our = 1 / 0.05  # 20Hz
    i=0

    # load imu data
    meta =  np.load(os.path.join(dataset_dir , sequence, 'dataset_info.npz'), allow_pickle=True)['meta'].item()
    imu_left = meta['imu']['/prophesee/left/imu']

    timestamps_imu = []
    for imu in imu_left:
        timestamps_imu.append(imu['ts'] - t_ref_events)\
        
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    while True:
        t = (i + 0.5)/est_freq_our
        i = i + 1
        # if i >100:  # for debug
        #     break
        print('t',t,'i',i)

        if t>timestamps_masks[len(timestamps_masks)-1]:
            break
        gt_frame_id = np.argmin(np.abs(np.array(timestamps_masks) - t))
        sys.stdout.flush()
        mask = mask_npz[mask_names[gt_frame_id]]
        # depth = gt_depth_npz[gt_depth_names[gt_frame_id]]
        object_ids = get_object_ids(mask)
        fx = 519.638  # in meta
        fy = 519.383972
        px = 321.661011
        py = 240.727005

        imu_idx = np.argmin(np.abs(np.asarray(timestamps_imu) - t))
        # imu_angular_vel = torch.as_tensor([x_vel[t], y_vel[t], z_vel[t]], dtype=torch.float, device=device)
        x, y, z = imu_left[imu_idx]['angular_velocity']['x'], imu_left[imu_idx]['angular_velocity']['y'], \
            imu_left[imu_idx]['angular_velocity']['z']
        #angular_vel = torch.as_tensor([y, -x, z], dtype=torch.float, device=device)
        imu_angular_vel = torch.as_tensor([y, -x, z], dtype=torch.float, device=device)
        imu_angular_vel2 = [y, -x, z]




        t_start = t - 0.5/est_freq_our
        t_end = t + 0.5/est_freq_our
        print('t_start',t_start,'t_end',t_end)
        idx_start = np.argmin(np.abs(events_proc[:, 0] - t_start))
        idx_end = np.argmin(np.abs(events_proc[:, 0] - t_end))
        print('event num',idx_end - idx_start)
        events_batch = deepcopy(events_proc[idx_start: idx_end])  # shape (32568, 4)  size 130272
        masked_batch = deepcopy(obj_mask[idx_start: idx_end])

        t_ref = events_proc[idx_start, 0]
        # print('events_batch',events_batch)
        events_batch[:, 0] = events_batch[:, 0] - t_ref
        # print('events_batch',events_batch)
        events_batch = events_batch[masked_batch]  # events_batch smaller here

        ALIGN = Estimator(height=height, width=width, fx=fx, fy=fy, px=px, py=py)

        if events_batch.shape[0] == 1:#  32568
            print('events_batch.shape[0] == 1')
            quit()

        events_origin = torch.as_tensor(events_batch, dtype=torch.float, device=device)
        events_origin[:, 1:] = ALIGN.warp_event(imu_angular_vel, events_origin)[:, 1:]

        events_show = torch.as_tensor(events_origin, dtype=torch.float, device=device)
        frame_orig = events2frame(events_show[: , :],fixed_size=True)
        frame = convGaussianFilter(frame_orig)
        event_img_orig = frame.sum(axis=0).cpu().detach().numpy()


        rot_vel = get_rot_vel(txt_file, t, est_freq_our, object_id)
        # rot_vel = rot_vel + imu_angular_vel2
        print('rot_vel',rot_vel)
        # for idx in range(len(events_batch)):
        #     event = events_batch[idx]
        #     warped_x, warped_y = get_warped_cent(event[1:3], rot_vel, fx, fy, px, py, event[0])
        #     # print('event[1:3]',event[1:3],'warped',(warped_x, warped_y))
        #     events_batch[idx, 1:3] = [warped_x, warped_y]

        events_origin_np = events_origin.cpu().numpy()
        for idx in range(len(events_origin_np)):
            event = events_origin_np[idx]
            warped_x, warped_y = get_warped_cent(event[1:3], rot_vel, fx, fy, px, py, event[0])
            # print('event[1:3]',event[1:3],'warped',(warped_x, warped_y))
            events_origin_np[idx, 1:3] = [warped_x, warped_y]


        # device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        events_show = torch.as_tensor(events_batch, dtype=torch.float, device=device)
        frame = events2frame(events_show[: , :],fixed_size=True)

        frame = convGaussianFilter(frame)
        event_img = frame.sum(axis=0).cpu().detach().numpy()

        # event_img = frame.sum(axis=0)
        sys.stdout.flush()

        # plt.figure()
        plt.figure(figsize=(10,5))
        plt.subplot(1, 2, 1)
        plt.title('rotated events')
        plt.imshow(event_img, cmap='bwr')
        plt.clim(-4, 4)
        plt.subplot(1, 2, 2)
        plt.title('orig. events')
        plt.imshow(event_img_orig, cmap='bwr')
        plt.clim(-4, 4)
        plt.show()

