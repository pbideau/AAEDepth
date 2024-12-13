import numpy as np
import time
from math import inf, nan
from copy import deepcopy
import os
from Estimators.Estimator import Estimator
from visualize.visualize import plot_img_map 
from utils.load import load_dataset
import torch
from torch import optim
from utils.utils import *
from scipy import stats
import math
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import re


def get_frame_by_index(frames, index):
    frames_names = list(frames.keys())
    frames_names.sort()
    frame_name = frames_names[index]
    return np.copy(frames[frame_name]) # To extract and keep in RAM

        
class VelocityEstimator(Estimator):
    def __init__(self, fx, fy, px, py, dataset_path, sequence, Ne, height, width, dt=0.025, overlap=0, fixed_size = True, padding = 100,
                    optimizer = 'Adam', optim_kwargs = None, lr = 0.05, lr_step = 250, lr_decay = 0.1, iters = 250, sigma_prior = 1,
                    version=0, gamma_param = [0.1, 1.59]
                    ) -> None:
        #_, fx, fy, px, py = load_dataset(dataset_path, sequence, width, height)
        super().__init__(height, width, fx, fy, px, py)
        self.dataset_path = dataset_path
        self.Ne = Ne
        self.dt = dt
        self.sequence = sequence
        self.overlap = overlap
        self.fixed_size = fixed_size
        self.padding = padding
        self.estimated_val = []
        self.img = []
        self.map = []
        self.time_record = []
        self.count = 1
        self.optimizer_name = optimizer
        self.optim_kwargs = optim_kwargs
        self.lr = lr
        self.lr_step = lr_step
        self.lr_decay = lr_decay
        self.iters = iters
        self.sigma_prior = sigma_prior
        self.version = version
        self.gamma_param = torch.tensor(gamma_param)
        print("Sequence: {}".format(sequence))
        self.loss_results = []

    @timer
    def __call__(self, events_set, save_filepath, *args, count = 1, save_figs = True, use_prev = True, rot_direction_filepath='') -> None:
        Ne = self.Ne
        overlap = self.overlap
        #para0 = np.array([0, 0, 0])
        para0 = np.array([0.0])
        res_device = np.array([0, 0, 0])
        self.count = count
        align_global = True

        obj_id = 0
        filename = os.path.basename(save_filepath)
        pattern = r'_(\d{8})\.txt$'
        match = re.search(pattern, filename)
        if match:
            obj_id = int(match.group(1))
        # else:
        #     raise ValueError(f"Object ID not found in the filename: {filename}")
        
        print(f"Extracted obj_id: {obj_id}")



        if os.path.isfile(rot_direction_filepath):
            align_global = False

        if args:
            masked = args[0]

        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        events_tensor_prev_aligned = 0

        while True:
            # print('self.count',self.count)
            # if self.count >1 :
            #     break
            start_time = time.time()

            if overlap:
                events_batch = deepcopy(events_set[int(Ne * (self.count - 1) * overlap): int(Ne + Ne * (self.count - 1) * overlap)])
            else:
                #events_batch = deepcopy(self.events_set[Ne * (self.count - 1): Ne * self.count])
                events_batch, masked_batch, t_end = self.sliceEvents_byTime(events_set, masked)

            #if len(events_batch) < Ne:
            if t_end > events_set[-1, 0]:
                break

            #events_batch_orig = events_batch
            t_ref = events_batch[0][0]
            t_end = events_batch[-1][0]

            events_batch[:, 0] = events_batch[:, 0] - t_ref
            events_batch = events_batch[masked_batch]
            events_batch = torch.from_numpy(events_batch).type(torch.float32).to(device)

            # read initialization of rotation if global estimates exist
            if align_global == False:
                results = pd.read_csv(rot_direction_filepath, sep=' ', header=None)
                results.columns = ['iter', 'ref_time', 'end_time', 'loss', 'x', 'y', 'z']
                para0 = np.array([results.iloc[self.count-1]['x'], results.iloc[self.count-1]['y'], 0])
                para0_tensor = torch.from_numpy(para0.copy()).float().to(device)
                print(para0)

                print('Number of Events: {}'.format(events_batch.shape[0]))
                print('{}: {}'.format(self.count, t_ref))

            if events_batch.shape[0] > 5: # and self.width*1/3 < x_med[0] < self.width*2/3 and self.height*1/3 < y_med[0] < self.height*2/3:

                if align_global==False:
                    # optimize magnitude of rotation
                    dt_prev = 0
                    res, loss = self.optimization(np.array([1.0]), events_batch, events_tensor_prev_aligned, para0_tensor, device, align_global,obj_id, dt_prev)
                    res = para0 * res
                else:
                    # optimize rotational direction (integrate over magnitude)
                    dt_prev = 0
                    res, loss = self.optimization(para0, events_batch, events_tensor_prev_aligned, res_device, device,
                                                  align_global,obj_id, dt_prev)
                    if self.count == 1:
                        para0_inv = np.array([math.pi])
                        res_inv, loss_inv = self.optimization(para0_inv, events_batch, events_tensor_prev_aligned, res_device, device, align_global,obj_id, dt_prev)
                        if loss_inv < loss:
                            res = res_inv
                            loss = loss_inv
                #dt_prev = torch.max(events_tensor[:, 0])

                # update new initial guess for next frame
                if use_prev:
                    para0 = res

                #self.estimated_val.append(np.append([self.count, t_ref.cpu().detach().numpy(), t_end.cpu().detach().numpy(), loss], res))
                if align_global==True:
                    #res = np.array([np.cos(res)[0], np.sin(res)[0], 0])
                    res = np.array([np.sin(res)[0], np.cos(res)[0], 0])
                    #res = np.append(res, 0)
                self.estimated_val.append(np.append([self.count, t_ref, t_end, loss], res))

                if save_figs:
                    img_name = '_' + f'{self.count:05d}' + '.png'
                    img_path = save_filepath.replace('.txt', img_name)
                    _, _, img_0, map_0 = self.calResult(events_batch, 0, np.array([0. ,0. ,0.]), warp=False)
                    _, _, img_1, map_1 = self.calResult(events_batch, 0,  res, warp=True)
                    clim = 4 if 'shapes' not in self.sequence else 10
                    cb_max = 8 if 'shapes' not in self.sequence else 20
                    plot_img_map([img_0, img_1],[map_0, map_1], clim, cb_max, filepath=img_path, save=True)

                #res_device = torch.from_numpy(res).float().to(device)
                #warped_events_batch = self.warp_event(res_device, events_tensor)
                #events_tensor_prev_aligned = warped_events_batch #self.warp_event(-res_device, warped_events_batch, rot_back=True)

                np.savetxt(save_filepath, np.array(self.estimated_val), fmt=[
                    '%d', '%.9f', '%.9f', '%.9f', '%.9f', '%.9f', '%.9f'], delimiter=' ')

            else:
                events_tensor_prev_aligned = 0
                dt_prev = 0

            self.count += 1
            duration = time.time() - start_time
            print("Duration:{}s\n".format(duration))


    def optimization(self, init_poses, events_tensor, events_tensor_prev_aligned, prev_rot, device, align_global,obj_id, dt_prev=0):
        # initializing local variables for class atrributes
        optimizer_name = self.optimizer_name
        optim_kwargs = self.optim_kwargs
        lr = self.lr
        lr_step = self.lr_step
        lr_decay = self.lr_decay
        iters = self.iters
        if not optim_kwargs:
            optim_kwargs = dict()
        if lr_step <= 0:
            lr_step = max(1, iters)
        
        # preparing data and prameters to be trained
        if init_poses is None:
            init_poses = np.zeros(1, dtype=np.float64)

        poses = torch.from_numpy(init_poses.copy()).float().to(device)
        poses.requires_grad = True

        # initializing optimizer
        optimizer = optim.__dict__[optimizer_name](
            [poses],lr =lr,amsgrad = True, **optim_kwargs)
        #scheduler = optim.lr_scheduler.StepLR(optimizer, lr_step, lr_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2*iters)
        print_interval = 1
        min_loss = inf
        best_poses = poses
        best_it = 0


        # optimization process
        if optimizer_name == 'Adam':
            for it in range(iters):
                optimizer.zero_grad()
                poses_val = poses.cpu().detach().numpy()
                if nan in poses_val:
                    print("nan in the estimated values, something wrong takes place, please check!")
                    exit()

                if align_global == False:
                    loss = self.loss_func(abs(poses), events_tensor, events_tensor_prev_aligned, align_global, self.sigma_prior, dt_prev, prev_rot)
                else:
                    loss = self.loss_func(poses, events_tensor, events_tensor_prev_aligned, align_global, self.sigma_prior, dt_prev, prev_rot)


                if loss < min_loss:
                    if init_poses.size == 1:
                        if align_global == False:
                            best_poses = abs(poses)
                        else:
                            best_poses = poses
                    else:
                        best_poses = poses
                    min_loss = loss.item()
                    best_it = it
                try:
                    loss.backward()
                except Exception as e:
                    print(e)
                    return poses_val, loss.item()
                optimizer.step()
                scheduler.step()
                #print(scheduler.get_lr())
        else:
            print("The optimizer is not supported.")

        best_poses = best_poses.cpu().detach().numpy()
        print('[Final Result]\tloss: {:.12f}\tposes: {} @ {}'.format(min_loss, best_poses, best_it))
        if device == torch.device('cuda:0'):
            torch.cuda.empty_cache()
        return best_poses, min_loss

    def sliceEvents_byTime(self, events, masked):

        t_start = events[0, 0] + self.dt * (self.count-1)
        t_end = events[0, 0] + self.dt * self.count

        #idx_start = torch.nonzero(events[:, 0] >= t_start)[0]
        #idx_end = torch.nonzero(events[:, 0] < t_end)[-1]
        #idx_start = torch.argmin(torch.abs(events[:, 0] - t_start))
        #idx_end = torch.argmin(torch.abs(events[:, 0] - t_end))
        #idx_start = np.where(events[:, 0] >= t_start)[0][0]
        #idx_end = np.where(events[:, 0] < t_end)[0][-1]

        idx_start = np.argmin(np.abs(events[:, 0] - t_start))
        idx_end = np.argmin(np.abs(events[:, 0] - t_end))

        events_batch = deepcopy(events[idx_start: idx_end])
        masked_batch = deepcopy(masked[idx_start: idx_end])

        return events_batch, masked_batch, t_end

    def create_circular_mask(self, h, w, center, radius):

        if center is None:  # use the middle of the image
            center = (int(w / 2), int(h / 2))
        if radius is None:  # use the smallest distance between the center and image walls
            radius = min(center[0], center[1], w - center[0], h - center[1])

        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

        mask = dist_from_center <= radius
        return mask