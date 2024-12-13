import torch
import numpy as np
import os

def test_camerapara():
    path_dataset = '/Users/cainan/Desktop/active_perception/semester_poster/EVIMO/data/evimo_npz/samsung_mono/sfm/eval'
    # sequences = os.listdir(path_dataset)
    sequences = [sequence for sequence in os.listdir(path_dataset) if not sequence.startswith('.DS_Store')]
    sequences = sequences[1:2]
    for sequence in sequences:
        meta = np.load('{}/{}/dataset_info.npz'.format(path_dataset, sequence), allow_pickle=True)['meta'].item()
        meta_ = meta['meta']
        fx = meta_['fx']
        fy = meta_['fx']
        px = meta_['cx']
        py = meta_['cy']
        dist_co = np.array([0.0, 0.0, 0.0, 0.0])
        dist_co[0] = meta_['k1']
        dist_co[1] = meta_['k2']
        dist_co[2] = meta_['p1']
        dist_co[3] = meta_['p2']
        instrinsic_matrix = np.array([[fx, 0, px], [0, fy, py], [0, 0, 1]])
        print('Sequence:', sequence)
        print('instrinsic_matrix:', instrinsic_matrix)
        print('dist_co:', dist_co)
        print('---')


if __name__ == '__main__':
    test_camerapara()
