#!/usr/bin/python

from __future__ import print_function

import numpy as np
import torch.utils.data as data

import utils


class heter_data(data.Dataset):
    def __init__(self, filename, transform=None, target_transform=None, target="hot"):
        """
            Args:
                filename: a list of pickle files.
                transform: transform applied to the feature data.
                target_transform: transform applied to the label data.
                target: target label encoding approach.
            Notes:
                Input data with key values: "feature", "label" (already one-hot encoded), "user"(optional).
                Feature with shape [seq_len, 8*2,  interval_len] ([12, 16, 7])
                For a single file we have:
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
            else:   # Train classifier, don't need one-hot encoding
                self.targets.extend(np.argmax(data["label"], axis=1))
        self.targets = np.array(self.targets)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.data)

    
class heter_data2(data.Dataset):
    def __init__(self, filename, transform=None, target_transform=None, target="hot"):
        self.transform = transform
        self.target_transform = target_transform
        self.data = []
        self.targets = []
        self.targets_t = []

        for file in filename:
            data_tmp = utils.load_pickle(file)
            self.data.extend(data_tmp["feature"])
            if target == "hot":  # is one-hot required?
                self.targets.extend(data_tmp["label"])
            else:  # Train classifier, don't need one-hot encoding
                self.targets.extend(np.argmax(data_tmp["label"], axis=1))
            self.targets_t.extend(data_tmp["label_t"])

        self.targets = np.array(self.targets)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target, target_t) 
            target is index of the global target class.
            target is index of the local target class.            
        """
        img, target, target_t = self.data[index], self.targets[index], self.targets_t[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
            target_t = self.target_transform(target_t)

        return img, target, target_t

    def __len__(self):
        return len(self.data)    
