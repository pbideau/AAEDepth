from Estimators.VelocityEstimator import VelocityEstimator
from utils.utils import *
        
class AngularVelocityEstimator(VelocityEstimator):
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
        self.trans_type = "rot"
    
    # def warp_event(self, poses, events, dt_prev=0):
    #     angular_vel_matrix = self.angular_velocity_matrix(poses)
    #     point_3d = self.events_form_3d_points(events)
    #     dt = events[:, 0]
    #     if dt_prev>0:
    #         dt[:] = dt_prev
    #     point_3d_rotated = torch.matmul(point_3d, angular_vel_matrix)
    #     r = torch.mul(torch.t(dt.repeat(3, 1)), point_3d_rotated)
    #     coordinate_3d = point_3d - r
    #     warped_x, warped_y = self.events_back_project(coordinate_3d)
    #     warped_events = torch.stack((dt, warped_x, warped_y, events[:, 3]), dim=1)
    #     return warped_events.squeeze()
    

    def warp_event(self, poses, events):
        # Check if poses is a batch
        if len(poses.shape) == 2:
            # warped_events_list = []
            # for pose in poses:
            #     angular_vel_matrix = self.angular_velocity_matrix(pose)
            #     point_3d = self.events_form_3d_points(events)   # n*3 
            #     dt = events[:, 0]
            #     point_3d_rotated = torch.matmul(point_3d, angular_vel_matrix)
            #     r = torch.mul(torch.t(dt.repeat(3, 1)), point_3d_rotated)
            #     coordinate_3d = point_3d - r
                
            #     warped_x, warped_y = self.events_back_project(coordinate_3d)
            #     warped_events = torch.stack((dt, warped_x, warped_y, events[:, 3]), dim=1)
            #     warped_events_list.append(warped_events)
            
            # # Stack all warped events
            # stacked_warped_events = torch.stack(warped_events_list)
            # return stacked_warped_events.squeeze()


            angular_vel_matrices = torch.stack([self.angular_velocity_matrix(pose) for pose in poses])
            point_3d = self.events_form_3d_points(events)  # n*3
            dt = events[:, 0]
            dt_repeated = dt.repeat(3, 1).unsqueeze(0).repeat(poses.shape[0], 1, 1)  # batch_size x 3 x n
            point_3d_rotated = torch.matmul(point_3d.unsqueeze(0), angular_vel_matrices)  # batch_size x n x 3
            r = torch.mul(dt_repeated.permute(0, 2, 1), point_3d_rotated)  # batch_size x n x 3
            coordinate_3d = point_3d.unsqueeze(0) - r  # batch_size x n x 3

            warped_x, warped_y = self.events_back_project(coordinate_3d.view(-1, 3))  # Flatten for back projection
            warped_x = warped_x.view(poses.shape[0], -1)  # Reshape back to batch_size x n
            warped_y = warped_y.view(poses.shape[0], -1)  # Reshape back to batch_size x n
            warped_events = torch.stack((dt.unsqueeze(0).repeat(poses.shape[0], 1), warped_x, warped_y, events[:, 3].unsqueeze(0).repeat(poses.shape[0], 1)), dim=2)
            return warped_events.squeeze()

        else:
            # Original single pose case
            angular_vel_matrix = self.angular_velocity_matrix(poses)  # 3*3
            point_3d = self.events_form_3d_points(events)  # n*3
            dt = events[:, 0]
            point_3d_rotated = torch.matmul(point_3d, angular_vel_matrix)  # n*3
            r = torch.mul(torch.t(dt.repeat(3, 1)), point_3d_rotated)
            coordinate_3d = point_3d - r
            
            warped_x, warped_y = self.events_back_project(coordinate_3d)
            warped_events = torch.stack((dt, warped_x, warped_y, events[:, 3]), dim=1)
            return warped_events.squeeze()

