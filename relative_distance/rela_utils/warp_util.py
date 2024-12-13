# import numpy as np
import torch

def angular_velocity_matrix(poses):
    # device = poses.device
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    if poses.shape[0] == 3:
        a1, a2, a3 = poses
    else:
        a1, a2 = poses
        a3 = torch.zeros(1, 1, dtype=torch.float, device=device)
    ang_vel_matrix = torch.zeros(3, 3, dtype=torch.float, device=device)
    ang_vel_matrix
    ang_vel_matrix[0, 1] = -a3
    ang_vel_matrix[0, 2] = a2
    ang_vel_matrix[1, 0] = a3
    ang_vel_matrix[1, 2] = -a1
    ang_vel_matrix[2, 0] = -a2
    ang_vel_matrix[2, 1] = a1
    return ang_vel_matrix

def events_forward_project(point,fx,fy,px,py):
    x = (point[0] - px) / fx
    y = (point[1] - py) / fy
    return x, y

def events_back_project(coordinate_3d,fx,fy,px,py):
    warped_x = coordinate_3d[0, 0] * fx / coordinate_3d[0, 2] + px
    warped_y = coordinate_3d[0, 1] * fy / coordinate_3d[0, 2] + py
    return warped_x, warped_y

def get_warped_cent(cent,rot_vel,fx,fy,px,py,dt):
    # print('getting warped cent')
    angular_vel_matrix = angular_velocity_matrix(rot_vel)
    # point_3d = events_form_3d_points(cent)
    X, Y = events_forward_project(cent,fx,fy,px,py)
    Z = 1
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    Z = torch.tensor(Z)

    # point_3d = torch.stack((X, Y, Z), dim=1)
    point_3d = torch.stack((X, Y, Z))
    # print("point_3d",point_3d)
    # dt = 0.02  # is the ESTIMATION_FREQ_OUR
    # dt =torch.tensor( 1 / est_freq_our)
    dt = torch.tensor(dt)
    # print('dt',dt)
    point_3d_rotated = torch.matmul(point_3d, angular_vel_matrix.double())
    r = torch.mul(torch.t(dt.repeat(3, 1)), point_3d_rotated)
    coordinate_3d = point_3d - r
    warped_x, warped_y = events_back_project(coordinate_3d,fx,fy,px,py)
    return warped_x,warped_y

