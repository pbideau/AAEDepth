import numpy as np
import pandas as pd
import cv2

def undo_distortion(src, instrinsic_matrix, distco=None):
    dst = cv2.undistortPoints(src, instrinsic_matrix, distco, None, instrinsic_matrix)
    return dst
    
def load_dataset(dataset_name, path_dataset, sequence):
    if dataset_name == "DAVIS_240C":
        calib_data = np.loadtxt('{}/{}/calib.txt'.format(path_dataset,sequence))

        fx = calib_data[0]
        fy = calib_data[1]
        px = calib_data[2]
        py = calib_data[3]
        dist_co = calib_data[4:]
        instrinsic_matrix = np.array([[fx, 0, px], [0, fy, py], [0, 0, 1]])

        width = 240
        height = 180

    if dataset_name == "evimo" or dataset_name == "evimo_ll" or dataset_name == "scioi_event":

        meta = np.load('{}/{}/dataset_info.npz'.format(path_dataset,sequence), allow_pickle=True)['meta'].item()

        if bool(dataset_name == "evimo") | bool(dataset_name == "evimo_ll"):
            meta = meta['meta']

        fx = meta['fx']
        fy = meta['fy']
        px = meta['cx']
        py = meta['cy']
        dist_co = np.array([0.0, 0.0, 0.0, 0.0])
        dist_co[0] = meta['k1']
        dist_co[1] = meta['k2']
        dist_co[2] = meta['p1']
        dist_co[3] = meta['p2']
        instrinsic_matrix = np.array([[fx, 0, px], [0, fy, py], [0, 0, 1]])

        width = meta['res_x']
        height = meta['res_y']

    LUT = np.zeros([width, height, 2])
    for i in range(width):
        for j in range(height):
            LUT[i][j] = np.array([i, j])
    LUT = LUT.reshape((-1, 1, 2))
    LUT = undo_distortion(LUT, instrinsic_matrix, dist_co).reshape((width, height, 2))

    return LUT, fx, fy, px, py, width, height
