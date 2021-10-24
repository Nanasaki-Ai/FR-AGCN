import os
import numpy as np
from numpy.lib.format import open_memmap

sets = {
    'train', 
}

datasets = {
    'ntu120/xsub',
}

from tqdm import tqdm

for dataset in datasets:
    for set1 in sets:
        print(dataset, set1)
        data = np.load('../data/{}/{}_data_joint.npy'.format(dataset, set1))
        N, C, T, V, M = data.shape
        T1 = T // 4
        forward = open_memmap(
            '../data/{}/{}_data_forward.npy'.format(dataset, set1),
            dtype='float32',
            mode='w+',
            shape=(N, 3, T1, V, M))
        forward[:, :, :T1, :, :] = data[:, :, ::4, :, :]