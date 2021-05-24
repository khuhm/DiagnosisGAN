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
        self.phase_list = {}
        for case in cases:
            self.phase_list[case] = sorted([file for file in all_files if file[7:10] == case[7:10]])

        self.input_phases = []
        for i, case in enumerate(cases):
            phases = sorted([file for file in all_files if file[7:10] == case[7:10]])
            combs = list(combinations(phases, num_phase))
            for comb in combs:
                self.input_phases.append(list(comb))

        self.num_comb = len(combs)

    def __len__(self):
        return len(self.input_phases)

    def __getitem__(self, idx):
        input_phases = self.input_phases[idx]
        case = input_phases[0][:6] + '0' + input_phases[0][7:10]
        label = self.labels[case]
        target_comb = idx % self.num_comb
        all_phases = self.phase_list[case]
        target_indices = [int(phase[6]) for phase in all_phases if phase not in input_phases]

        imgs = []
        segs = []
        for phase in all_phases:
            imgs.append(np.transpose(np.load(os.path.join(self.case_dir, phase))).astype(np.float32))
            segs.append(np.transpose(np.load(os.path.join(self.seg_dir, phase))))

        imgs = np.stack(imgs)
        segs = np.stack(segs)
        slice_margin = np.array(self.crop_size) - np.array(imgs.shape[1:])
        padding = ((0, 0),
                   (slice_margin[0] // 2, slice_margin[0] - slice_margin[0] // 2),
                   (slice_margin[1] // 2, slice_margin[1] - slice_margin[1] // 2),
                   (slice_margin[2] // 2, slice_margin[2] - slice_margin[2] // 2),)
        imgs = np.pad(imgs, padding, 'constant')
        segs = np.pad(segs, padding, 'constant')

        imgs_four = np.copy(imgs)
        segs_four = np.copy(segs)

        imgs[target_indices] = 0
        syn_in = []
        for target_idx in target_indices:
            phase_mask = np.zeros(imgs.shape, dtype=np.float32)
            phase_mask[target_idx] = 1
            syn_in.append(np.concatenate((imgs, phase_mask), axis=0))

        if syn_in:
            syn_in = np.stack(syn_in)

        syn_target = imgs_four[target_indices]

        return {'img': imgs,
                'seg': segs,
                'imgs_four': imgs_four,
                'segs_four': segs_four,
                'syn_in': syn_in,
                'syn_target': syn_target,
                'target_comb': target_comb,
                'target_indices': target_indices,
                'label': label}
