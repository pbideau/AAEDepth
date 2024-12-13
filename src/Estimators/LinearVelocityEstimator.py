from Estimators.VelocityEstimator import VelocityEstimator
from utils.utils import *
        
class LinearVelocityEstimator(VelocityEstimator):
    def __init__(self, fx, fy, px, py, dataset_path, sequence, Ne, height, width, dt=0.025, overlap=0, fixed_size = False, padding = 0,
                    optimizer = 'Adam', optim_kwargs = None, lr = 0.01, lr_step = 80, lr_decay = 0.1, iters = 80, sigma_prior=1, version=0, gamma_param = [0.1, 1.59]
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
        self.trans_type = "trans"

    def __call__(self, save_filepath, *args, count = 1, save_figs = True, use_prev = True):
        super().__call__(save_filepath, *args, count = count, save_figs = save_figs, use_prev = True)

    def warp_event(self, poses, events, dt_prev=0):
        linear_vel_vector = self.linear_velocity_vector(poses)
        point_3d = self.events_form_3d_points(events)
        dt = events[:, 0].unsqueeze(1)
        if dt_prev>0:
            dt[:] = dt_prev
        coordinate_3d = point_3d - dt * linear_vel_vector
        warped_x, warped_y = self.events_back_project(coordinate_3d)
        warped_events = torch.stack((dt.squeeze(), warped_x, warped_y, events[:, 3]), dim=1)
        return warped_events.squeeze()
