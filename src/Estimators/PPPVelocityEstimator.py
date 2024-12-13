from Estimators.VelocityEstimator import VelocityEstimator
from utils.utils import *
import matplotlib.pyplot as plt

class PPPVelocityEstimator(VelocityEstimator):
    def __init__(self, fx, fy, px, py, dataset_path, sequence, Ne, height, width, dt=0.025, overlap=0, fixed_size = True, padding = 100,
                    optimizer = 'Adam', optim_kwargs = None, lr = 0.05, lr_step = 250, lr_decay = 0.1, iters = 250, sigma_prior=1, version=0, gamma_param = [0.1, 1.59]
                    ) -> None:
        super().__init__(   fx,
                            fy,
                            px,
                            py,
                            dataset_path,
                            sequence,
                            Ne,
                            height,
                            width,
                            dt,
                            overlap,
                            fixed_size,
                            padding,
                            optimizer,
                            optim_kwargs,
                            lr,
                            lr_step,
                            lr_decay,
                            iters,
                            sigma_prior,
                            version,
                            gamma_param)
        self.method = "st-ppp"
        
    def loss_func(self, x, events_batch, events_tensor_prev_aligned, align_global, v=1, dt_prev=0, pose_prev=[0, 0, 0]) -> torch.float32:

        interval = 0.001
        if align_global == True:
            max_sum = 0.1
        else:
            max_sum = interval

        loss = 0

        if align_global == False:
            X = x*pose_prev
            warped_events_batch = self.warp_event(X, events_batch)   # warped_events_batch m*4*n
            # if dt_prev>0:
            #     print('dt_prev',dt_prev)
            #     quit()
            #     warped_events_batch = self.warp_event(pose_prev, warped_events_batch, dt_prev)

            frame = self.events2frame(warped_events_batch, fixed_size = self.fixed_size, padding = self.padding)
            frame = convGaussianFilter(frame, sigma=1)

            # if torch.is_tensor(events_tensor_prev_aligned):
            #     events_tensor_prev_aligned = self.events2frame(events_tensor_prev_aligned, fixed_size=self.fixed_size, padding=self.padding)
            #     events_tensor_prev_aligned = convGaussianFilter(events_tensor_prev_aligned, sigma = v)

            loss,_ = self.poisson(frame.abs(), events_tensor_prev_aligned)

            # loss = loss + loss_m
        else:
            for s in range(0, int(max_sum/interval)):
                X = torch.cat([s * interval * torch.cos(x), s * interval * torch.sin(x)] ) # 1*2 vector       n*2 vector
                warped_events_batch = self.warp_event(X, events_batch)  # warp a lot warped_events_batch at the same time
                # warped_events_batch m*4*n
                # for warped_events in warped_events_batch:    # warped_events m*4
                if dt_prev>0:
                    #print(pose_prev)
                    warped_events_batch = self.warp_event(pose_prev, warped_events_batch, dt_prev)

                frame = self.events2frame(warped_events_batch, fixed_size = self.fixed_size, padding = self.padding)
                frame = convGaussianFilter(frame, sigma=1)

                if torch.is_tensor(events_tensor_prev_aligned):
                    events_tensor_prev_aligned = self.events2frame(events_tensor_prev_aligned, fixed_size=self.fixed_size, padding=self.padding)
                    events_tensor_prev_aligned = convGaussianFilter(events_tensor_prev_aligned, sigma = v)

                loss_m,_ = self.poisson(frame.abs(), events_tensor_prev_aligned)
                # print('loss_m',loss_m)

                loss = loss + loss_m

            # s_values = torch.arange(0, max_sum, interval, device=x.device)
            # cos_x = torch.cos(x)
            # sin_x = torch.sin(x)
            # X = torch.stack((s_values * cos_x, s_values * sin_x), dim=1)  # n*2 matrix

            # warped_events_batch = self.warp_event(X, events_batch)  # warp a lot of events at the same time

            # if dt_prev > 0:
            #     warped_events_batch = self.warp_event(pose_prev, warped_events_batch, dt_prev)

            # frame = self.events2frame(warped_events_batch, fixed_size=self.fixed_size, padding=self.padding)
            # frame = convGaussianFilter(frame, sigma=1)

            # if torch.is_tensor(events_tensor_prev_aligned):
            #     events_tensor_prev_aligned = self.events2frame(events_tensor_prev_aligned, fixed_size=self.fixed_size, padding=self.padding)
            #     events_tensor_prev_aligned = convGaussianFilter(events_tensor_prev_aligned, sigma=v)

            # loss_m, _ = self.poisson(frame.abs(), events_tensor_prev_aligned)
            # print('loss_m',loss_m)
            # loss = loss_m.sum()  # Sum over the batch dimension




        return loss
    
    def calResult(self, events_batch, events_tensor_prev_aligned, para, warp = True, cal_loss = True):
        device = events_batch.device
        poses = torch.from_numpy(para).float().to(device)
        with torch.no_grad():
            if warp:
                warped_events_batch = self.warp_event(poses, events_batch)
            else:
                point_3d = self.events_form_3d_points(events_batch)
                warped_x, warped_y = self.events_back_project(point_3d)
                warped_events_batch = torch.stack((events_batch[:, 0], warped_x, warped_y, events_batch[:, 3]), dim=1)

            frame = self.events2frame(warped_events_batch, fixed_size=self.fixed_size, padding=self.padding)
            frame = convGaussianFilter(frame)
            img = frame.sum(axis=0).cpu().detach().numpy()

            if cal_loss:
                if torch.is_tensor(events_tensor_prev_aligned):
                    events_tensor_prev_aligned = self.events2frame(events_tensor_prev_aligned, fixed_size=self.fixed_size, padding=self.padding)
                    events_tensor_prev_aligned = convGaussianFilter(events_tensor_prev_aligned, sigma=1)
                loss, map = self.poisson(frame.abs(), events_tensor_prev_aligned)
                loss = loss.item()
                map = map.cpu().detach().numpy()

            else:
                loss = 0
                map = 0

            torch.cuda.empty_cache()

        return frame, loss, img, map