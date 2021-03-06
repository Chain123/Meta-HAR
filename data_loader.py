#!/usr/bin/python

from __future__ import print_function
import torch.nn.init as init
import torch.nn as nn
from torch.utils.model_zoo import tqdm
import torch.utils.data as data

import sys
import time
import math
import os
import hashlib
import errno

import numpy as np
import utils


class heter_data(data.Dataset):
    def __init__(self, filename, transform=None, target_transform=None, target="hot"):
        """ Input data with key values: "feature", "label" (one-hot), "user". 
            Feature with shape [num_sampels, seqp_len, 8*2,  interval_len]
        """
        self.transform = transform
        self.target_transform = target_transform
        self.data = []
        self.targets = []

        for file in filename:
            data = utils.load_pickle(file)
            self.data.extend(data["feature"])
            if target == "hot":  # is one-hot required?
                self.targets.extend(data["label"])
            else:  # Train classifier, don't need one-hot encoding
                self.targets.extend(np.argmax(data["label"], axis=1))
        self.targets = np.array(self.targets)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.data)
