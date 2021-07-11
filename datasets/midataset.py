import numpy as np
import os
import copy
import torch
import random
import utils as du
from torch.utils.data import Dataset


class SliceDataset(Dataset):
    def __init__(self, data_path, dataset='kits', task='tumor', series_index=None, train=True):
        super(SliceDataset, self).__init__()
        assert dataset in ['lits', 'kits']
        assert task in ['organ', 'tumor']
        self.load_path = data_path
        self.series_index = series_index
        self.task = task
        self.train = train
        self.dataset = dataset
        self.tumor_slices = np.load(os.path.join(data_path, '%s_%s_slices.npy' % (dataset, task)))

    def rotate(self, img, mask, k=None):
        if k is None:
            k = random.choice([0, 1, 2, 3])
        img = np.rot90(img, k, (-2, -1))
        mask = np.rot90(mask, k, (-2, -1))

        return img, mask

    def flip(self, img, mask, flip=None):
        if flip is None:
            a, b = random.choice([1, -1]), random.choice([1, -1])
        else:
            a, b = flip
        if img.ndim == 2:
            img = img[::a, ::b]
        elif img.ndim == 3:
            img = img[:, ::a, ::b]
        mask = mask[::a, ::b]

        return img, mask

    def __len__(self):
        return len(self.tumor_slices)

    def __getitem__(self, item):
        # Data loading
        f_name = self.tumor_slices[item]
        case = f_name.split('_')[0]
        npz = np.load(os.path.join(self.load_path, f_name), allow_pickle=True)
        ct = npz.get('ct')
        mask = npz.get('mask')

        # Preprocess
        if self.task == 'organ':
            mask[mask > 0] = 1
        elif self.task == 'tumor':
            mask = mask >> 1
            mask[mask > 0] = 1

        if self.dataset == 'lits':
            ct = du.window_standardize(ct, -60, 140)
        elif self.dataset == 'kits':
            ct = du.window_standardize(ct, -200, 300)

        if self.train:
            ct, mask = self.flip(ct, mask)
            ct, mask = self.rotate(ct, mask)

        # one-hot
        img0 = copy.deepcopy(mask)
        img0 += 1
        img0[img0 != 1] = 0
        mask = np.stack((img0, mask), axis=0)
        mask[mask > 0] = 1

        # To tensor & cut to 384
        ct = torch.from_numpy(du.cut_384(ct.copy())).unsqueeze(0).float()
        mask = torch.from_numpy(du.cut_384(mask.copy())).float()

        return ct, mask, case
