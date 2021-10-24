import os
import numpy as np
from numpy.lib.format import open_memmap

paris = {
    'ntu120/xsub': (
        (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
        (13, 1),
        (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18), (20, 19), (22, 23), (21, 21), (23, 8), (24, 25),
        (25, 12)
    ),
    'ntu120/xset': (
        (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
        (13, 1),
        (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18), (20, 19), (22, 23), (21, 21), (23, 8), (24, 25),
        (25, 12)
    ),
}

sets = {
    'train', 'val'
}

datasets = {
    'ntu120/xsub', 'ntu120/xset',
}
# bone
from tqdm import tqdm

for dataset in datasets:
    for set1 in sets:
        print(dataset, set1)
        data = np.load('../data/{}/{}_data_reverse.npy'.format(dataset, set1))
        N, C, T, V, M = data.shape
        fp_sp = open_memmap(
            '../data/{}/{}_data_reverse_bone.npy'.format(dataset, set1),
            dtype='float32',
            mode='w+',
            shape=(N, 3, T, V, M))

        for v1, v2 in tqdm(paris[dataset]):
            v1 -= 1
            v2 -= 1

            fp_sp[:, :, :, v1, :] = data[:, :, :, v1, :] - data[:, :, :, v2, :]
