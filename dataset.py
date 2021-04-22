from torch.utils.data import Dataset
import json
import os
import numpy as np
import random
from itertools import combinations


class CT(Dataset):
    def __init__(self,
                 mode='train',
                 data_dir=None,
                 num_phase=None,
                 crop_size=(96, 160, 192)):
        self.mode = mode
        self.case_dir = os.path.join(data_dir, 'image_npy')
        self.seg_dir = os.path.join(data_dir, 'segmentation_npy')
        self.crop_size = crop_size

        with open(os.path.join(data_dir, mode + '.txt'), 'r') as file:
            cases = json.load(file)

        with open(os.path.join(data_dir, 'subtypes.json')) as json_file:
            self.labels = json.load(json_file)

        all_files = sorted(os.listdir(self.case_dir))

        self.phases = []
        for i, case in enumerate(cases):
            phases = sorted([file for file in all_files if file[7:10] == case[7:10]])
            combs = list(combinations(phases, num_phase))
            for comb in combs:
                self.phases.append(list(comb))

        self.num_comb = len(combs)

    def __len__(self):
        return len(self.phases)

    def __getitem__(self, idx):
        phases = self.phases[idx]
        case = phases[0][:6] + '0' + phases[0][7:10]
        label = self.labels[case]
        target_idx = idx % self.num_comb

        imgs = []
        segs = []
        for phase in phases:
            imgs.append(np.transpose(np.load(os.path.join(self.case_dir, phase))).astype(np.float32))
            segs.append(np.transpose(np.load(os.path.join(self.seg_dir, phase))))

        imgs = np.stack(imgs)
        segs = np.stack(segs)
        slice_margin = np.array(self.crop_size) - np.array(imgs.shape[1:])
        padding = ((0, 0),
                   (slice_margin[0] // 2, slice_margin[0] - slice_margin[0] // 2),
                   (slice_margin[1] // 2, slice_margin[1] - slice_margin[1] // 2),
                   (slice_margin[2] // 2, slice_margin[2] - slice_margin[2] // 2),)
        imgs = np.pad(imgs, padding)
        segs = np.pad(segs, padding)

        return {'img': imgs,
                'seg': segs,
                'target_idx': target_idx,
                'label': label}
