from math import *
import torch
from utils.utils import  *
from visualize.visualize import *
from copy import deepcopy

class Estimator(object):

    eps = torch.finfo(torch.float32).eps

    def __init__(self, height=0, width=0, fx=0, fy=0, px=0, py=0) -> None:
        super().__init__()
        self.height = height
        self.width = width
        self.fx = fx
        self.fy = fy
        self.px = px
        self.py = py
    
    def angular_velocity_matrix(self, poses):
        #if poses.shape[0] == 3:
        # print("poses",poses)
        # print('poses.shape',poses.shape, )
        # if not isinstance(poses, list):
        if poses.shape[0] == 3:
            device = poses.device
            a1, a2, a3 = poses

            ang_vel_matrix = torch.zeros(3, 3, dtype=torch.float, device=device)
            ang_vel_matrix[0, 1] = -a3
            ang_vel_matrix[0, 2] = a2
            ang_vel_matrix[1, 0] = a3
            ang_vel_matrix[1, 2] = -a1
            ang_vel_matrix[2, 0] = -a2
            ang_vel_matrix[2, 1] = a1
            return ang_vel_matrix

        if poses.shape[0] == 2:
            device = poses[0].device
            a1 = poses[0]
            a2 = poses[1]
            a3 = torch.zeros(1, 1, dtype=torch.float, device=device)
            
            ang_vel_matrix = torch.zeros(3, 3, dtype=torch.float, device=device)
            ang_vel_matrix[0, 1] = -a3
            ang_vel_matrix[0, 2] = a2
            ang_vel_matrix[1, 0] = a3
            ang_vel_matrix[1, 2] = -a1
            ang_vel_matrix[2, 0] = -a2
            ang_vel_matrix[2, 1] = a1
            return ang_vel_matrix

        # if poses.shape[0] > 3:
        if len(poses.shape) == 2:
            batch_size = poses.shape[0]
            device = poses.device

            a1 = poses[:, 0]
            a2 = poses[:, 1]
            a3 = torch.zeros(batch_size, dtype=torch.float, device=device)

            ang_vel_matrices = torch.zeros(3, 3, batch_size, dtype=torch.float, device=device)
            ang_vel_matrices[0, 1, :] = -a3
            ang_vel_matrices[0, 2, :] = a2
            ang_vel_matrices[1, 0, :] = a3
            ang_vel_matrices[1, 2, :] = -a1
            ang_vel_matrices[2, 0, :] = -a2
            ang_vel_matrices[2, 1, :] = a1
            # ang_vel_matrices = ang_vel_matrices.permute(1, 2, 0)  # Stack along the 3rd dimension
            return ang_vel_matrices
        

    def linear_velocity_vector(self, poses, device='cuda'):
        #device = poses.device
        a1, a2, a3 = poses
        lin_vel_vector = torch.zeros(3, dtype=torch.float, device=device)
        lin_vel_vector[0] = a1
        lin_vel_vector[1] = a2
        lin_vel_vector[2] = a3
        return lin_vel_vector

    # def warp_event(self, poses, events):

    #     angular_vel_matrix = self.angular_velocity_matrix(poses)
    #     point_3d = self.events_form_3d_points(events)
    #     dt = events[:, 0]
    #     point_3d_rotated = torch.matmul(point_3d, angular_vel_matrix)
    #     r = torch.mul(torch.t(dt.repeat(3, 1)), point_3d_rotated)
    #     coordinate_3d = point_3d - r
        
    #     warped_x, warped_y = self.events_back_project(coordinate_3d)
    #     warped_events = torch.stack((dt, warped_x, warped_y, events[:, 3]), dim=1)

    #     return warped_events.squeeze()

    def warp_event(self, poses, events):
        # Check if poses is a batch
        if len(poses.shape) == 2:
            warped_events_list = []
            for pose in poses:
                angular_vel_matrix = self.angular_velocity_matrix(pose)  # 3*3*n
                point_3d = self.events_form_3d_points(events)   # m*3*n
                dt = events[:, 0]  # m TODO m*n
                point_3d_rotated = torch.matmul(point_3d, angular_vel_matrix)  # m *3 * n
                r = torch.mul(torch.t(dt.repeat(3, 1)), point_3d_rotated)  # TODO m*3*n  each* m*3*n = m*3*n
                coordinate_3d = point_3d - r  # m*3*n - m*3*n = m*3*n
                
                warped_x, warped_y = self.events_back_project(coordinate_3d)  # TODO m*n
                warped_events = torch.stack((dt, warped_x, warped_y, events[:, 3]), dim=1)  # m*n,m*n,m*n,TODO m*n
                # warped_events m*4*n
                warped_events_list.append(warped_events)
            
            # Stack all warped events
            stacked_warped_events = torch.stack(warped_events_list)
            return stacked_warped_events.squeeze()
        else:
            # Original single pose case    m events
            angular_vel_matrix = self.angular_velocity_matrix(poses)  # 3*3
            point_3d = self.events_form_3d_points(events)  # m*3
            dt = events[:, 0]  # m
            point_3d_rotated = torch.matmul(point_3d, angular_vel_matrix)  # m*3  * 3*3 = m*3
            r = torch.mul(torch.t(dt.repeat(3, 1)), point_3d_rotated)  #  !!m*3 each* m*3 = m*3
            coordinate_3d = point_3d - r  # m*3
            
            warped_x, warped_y = self.events_back_project(coordinate_3d)
            warped_events = torch.stack((dt, warped_x, warped_y, events[:, 3]), dim=1)
            return warped_events.squeeze()


    def events_back_project(self, coordinate_3d):  # m*3    m events  TODO m*3*n
        warped_x = coordinate_3d[:, 0] * self.fx / coordinate_3d[:, 2] + self.px # m  TODO m*n
        warped_y = coordinate_3d[:, 1] * self.fy / coordinate_3d[:, 2] + self.py # m
        return warped_x, warped_y

    def events_forward_project(self, events):      
        x = (events[:, 1] - self.px) / self.fx
        y = (events[:, 2] - self.py) / self.fy
        return x, y

    def events_form_3d_points(self, events):
        device = events.device
        N = events.shape[0]
        Z = torch.ones(N, dtype=torch.float, device=device)
        X, Y = self.events_forward_project(events)
        point_3d = torch.stack((X, Y, Z), dim=1)
        return point_3d

    def accumulate_event(self, events, fixed_size = False, **kwargs):
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
            iwe = self.accumulate_operation(xx, yy, dx, dy, polarity, H, W, device)
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
            iwe = self.accumulate_operation(xx, yy, dx, dy, polarity, H, W, device)
            return iwe[:-1, :-1]

    def accumulate_operation(self, pxx, pyy, dx, dy, polarity, height, width, device):
        iwe = torch.zeros(size = torch.Size([height, width]), device = device)
        iwe.index_put_((pyy    , pxx    ), polarity*(1.-dx)*(1.-dy), accumulate=True)
        iwe.index_put_((pyy    , pxx + 1), polarity*dx*(1.-dy), accumulate=True)
        iwe.index_put_((pyy + 1, pxx    ), polarity*(1.-dx)*dy, accumulate=True)
        iwe.index_put_((pyy + 1, pxx + 1), polarity*dx*dy, accumulate=True)
        return iwe
    
    def calResult(self, events_batch, para, *args,  warp = True, cal_loss = True, fixed_size = False, padding = 0):
        pass

    def events2frame(self, warped_events_batch, fixed_size = False, padding = 0):
        device = warped_events_batch.device
        #pos_batch = warped_events_batch[warped_events_batch[:,3]==1]
        #neg_batch = warped_events_batch[warped_events_batch[:,3]==0]
        if not fixed_size:
            min_x = torch.min(warped_events_batch[:, 1]).floor().long()
            max_x = torch.max(warped_events_batch[:, 1]).ceil().long()
            min_y = torch.min(warped_events_batch[:, 2]).floor().long()
            max_y = torch.max(warped_events_batch[:, 2]).ceil().long()
            new_height = torch.max(max_y-min_y+1, self.height-min_y)
            new_width = torch.max(max_x-min_x+1, self.width-min_x)
            
        else:
            min_x = torch.tensor(-padding, device = device)
            min_y = torch.tensor(-padding, device = device)
            new_height = torch.tensor(self.height + 2 * padding)
            new_width = torch.tensor(self.width + 2 * padding)

        if len(warped_events_batch.size())<=1:
            frame0 = torch.zeros(new_height, new_width, device=device)
            frame1 = torch.zeros(new_height, new_width, device=device)
        else:
            pos_batch = warped_events_batch[warped_events_batch[:,3]==1]
            neg_batch = warped_events_batch[warped_events_batch[:,3]==0]
            frame1 = self.accumulate_event(pos_batch, fixed_size=fixed_size, min_x = min_x, min_y = min_y, height=new_height, width=new_width)
            frame0 = self.accumulate_event(neg_batch, fixed_size=fixed_size, min_x = min_x, min_y = min_y, height=new_height, width=new_width)

        frame = torch.stack((frame1, frame0))
        return frame

    def loss_func(self, x, events_batch, *args) -> torch.float32:
        pass

    def variance(self, iwe, *args):
        return -iwe.std()**2, 0

    def poisson(self, iwe, events_tensor_prev_aligned):
        #r, beta = torch.tensor(args)
        r = self.gamma_param[0]
        beta = self.gamma_param[1]
        # update r and beta according to current observatio
        if torch.is_tensor(events_tensor_prev_aligned):
            beta = beta + 1
            r = r + events_tensor_prev_aligned.abs()
        #r = r + iwe

        map = ((iwe + r).lgamma() + r * (beta).log() - (iwe + 1.).lgamma() - (r).lgamma() - (iwe + r) * (beta + 1).log())
        if iwe[0,:,:].sum() != 0:
            l1 = map[0,:,:].sum()/iwe[0,:,:].sum()
        else:
            l1 = 0
        if iwe[1,:,:].sum() != 0:
            l2 = map[1,:,:].sum()/iwe[1,:,:].sum()
        else:
            l2 = 0
        loss = -(l1 + l2)

        return loss, -map.sum(0)
