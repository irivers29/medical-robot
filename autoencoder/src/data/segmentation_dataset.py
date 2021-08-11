"""Data utility functions."""
import os
from posix import POSIX_FADV_WILLNEED

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from torchvision import transforms
import os

import _pickle as pickle


LABELS_LIST = [
    {"id": -1, "name":"void", "rgb_values": [0, 0, 0]},

]


class Data(data.Dataset):

    def __init__(self, image_path):
        self.root_dir_name = os.path.dirname(image_path)

        print(os.path)
        
        with open(image_path) as f:
            self.image_names = f.read().splitlines()

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, key):
        if isinstance(key, slice):
            # get the start, stop, and step from the slice
            return [self[ii] for ii in range(*key.indices(len(self)))]
        elif isinstance(key, int):
            # handle negative indices
            if key < 0:
                key += len(self)
            if key < 0 or key >= len(self):
                raise IndexError("The index (%d) is out of range." % key)
            # get the data from direct index
            return self.get_item_from_index(key)
        else:
            raise TypeError("Invalid argument type.")
