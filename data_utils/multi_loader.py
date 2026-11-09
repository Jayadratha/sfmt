import numpy as np
import torch
from torch_geometric.data import Dataset, Data
from data_utils.mixamo_loader_vivek import MixamoDataset,MixamoValidationDataset
from data_utils.rignet_loader import RignetDataset
from data_utils.amass_loader import AmassDataset
import math

class MultiDataset(Dataset):
    # only used for training keypoint learning
    def __init__(self, amass_dir, mixamo_dir, rignet_dir,
                 smpl, part_augmentation=False, prob=(1/3, 1/3, 1/3), preload=None, single_part=True,
                 part_aug_scale=((0.5, 4), (0.6, 1), (0.3, 1.5)), simplify=True, new_rignet=True, seed=2023):
        # prob: probability of Amass data
        super(MultiDataset, self).__init__()
        
        # self.ind=0
        if preload:
            self.amass, self.mixamo, self.rignet = preload
            # self.amass, self.rignet = preload
        else:
            if isinstance(part_augmentation, bool):
                p1 = p2 = p3 = part_augmentation
            else:
                p1, p2, p3 = part_augmentation
            # AMASS / RigNet are only loaded when they can actually be sampled (prob > 0),
            # so Mixamo-only training (prob=(0, 1, 0)) does not need those datasets on disk.
            self.amass = AmassDataset(amass_dir, smpl, part_augmentation=p1, simplify=simplify) if prob[0] > 0 else None
            self.mixamo = MixamoDataset(mixamo_dir, flag='train', part_augmentation=p2, single_part=single_part, part_aug_scale=part_aug_scale)
            if prob[2] > 0:
                self.rignet = RignetDataset(rignet_dir, flag='humanoid_train_new' if new_rignet else 'humanoid_train')
            else:
                self.rignet = None
        # --- stateless data schedule -------------------------------------------------------------------------
        # Sample `index` of an epoch (batch size 2 -> samples 2k and 2k+1 form training iteration k) is a pure function of
        # (iteration, seed): even samples read pose k, odd samples pose len(poses)//2 + k; the characters change every 60
        # iterations (= one 60-frame motion) and are drawn from a RandomState seeded with (seed, block).  So data loader
        # workers can prepare batches in parallel, a resumed run sees exactly the same data, and every experiment sees
        # the same character schedule (before, it depended on how much global RNG state the losses had consumed).
        self.prob = prob
        self.seed = seed
        self.start_iter = 0          # iteration the current epoch starts at (!= 0 only when resuming mid-epoch)

    def chars_for_block(self, block):
        n = len(self.mixamo.vs)
        rs = np.random.RandomState([self.seed, int(block)])
        c0, c1 = rs.randint(n), rs.randint(n)
        c2 = rs.choice([i for i in range(n) if i != c0])
        c3 = rs.choice([i for i in range(n) if i != c1])
        return c0, c1, c2, c3

    def len(self):
        return len(self.mixamo)

    def database(self):
        return self.amass, self.mixamo, self.rignet

    def reset(self, y=0):            # kept for compatibility; the schedule is stateless now
        pass

    def set_index(self, y=0):        # kept for compatibility; the schedule is stateless now
        pass

    def get(self, index):
        k = self.start_iter + index // 2                      # training iteration this sample belongs to
        c0, c1, c2, c3 = self.chars_for_block(k // 60)
        if index % 2 == 0:
            return self.mixamo.get_uniform(k, c0, c2)
        return self.mixamo.get_uniform(len(self.mixamo.poses) // 2 + k, c1, c3)
