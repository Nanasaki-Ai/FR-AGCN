import os
import numpy as np
from numpy.lib.format import open_memmap

paris = {
    'uav': (
    (10, 8), (8, 6), (9, 7), (7, 5), # arms
    (15, 13), (13, 11), (16, 14), (14, 12), # legs
    (11, 5), (12, 6), (11, 12), (5, 6), # torso
    (5, 0), (6, 0), (1, 0), (2, 0), (3, 1), (4, 2) # nose, eyes and ears
    ),
}

sets = {
    'train', 'test',
}

datasets = {
    'uav',
}
# bone
from tqdm import tqdm

for dataset in datasets:
    for set1 in sets:
        print(dataset, set1)
        print('generate forward bone data')
        data = np.load('E:/PyTorchTest/TEST/UAVAGCN/data/{}/{}_data_forward.npy'.format(dataset, set1))
        N, C, T, V, M = data.shape
        fp_sp = open_memmap(
            'E:/PyTorchTest/TEST/UAVAGCN/data/{}/{}_data_forward_bone.npy'.format(dataset, set1),
            dtype='float32',
            mode='w+',
            shape=(N, 3, T, V, M))

        for v1, v2 in tqdm(paris[dataset]):
            v1 -= 1
            v2 -= 1

            fp_sp[:, :, :, v1, :] = data[:, :, :, v1, :] - data[:, :, :, v2, :]
