import os
import sys

sys.path.extend(['../'])
import numpy as np
from numpy.lib.format import open_memmap



sets = {
    'train', 'val',
}

# 'ntu/xview', 'ntu/xsub'
datasets = {
    'uav',
}

from tqdm import tqdm

for dataset in datasets:
    for set1 in sets:
        print(dataset, set1)
        data = np.load('E:/PyTorchTest/TEST/UAVAGCN/data/{}/{}_data.npy'.format(dataset, set1))
        N, C, T, V, M = data.shape
        T1 = T // 2
        forward = open_memmap(
            'E:/PyTorchTest/TEST/UAVAGCN/data/{}/{}_data_forward.npy'.format(dataset, set1),
            dtype='float32',
            mode='w+',
            shape=(N, 3, T1, V, M))
        forward[:, :, :T1, :, :] = data[:, :, ::2, :, :]       