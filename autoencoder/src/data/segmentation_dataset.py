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
    {"id": 0, "name":"PIP", "rgb_values": [36, 28, 236]},
    {"id": 1, "name":"MCP", "rgb_values": [69, 209, 14]},
    {"id": 2, "name":"CMC", "rgb_values": [183, 50, 250]},
    {"id": 3, "name":"Wrist", "rgb_values": [10, 0, 0]}
]

def label_img_to_rgb(label_img):
    label_img = np.squeeze(label_img)
    labels = np.unique(label_img)
    label_infos = [l for l in LABELS_LIST if l['id'] in labels]

    label_img_rgb = np.array([label_img,
                              label_img,
                              label_img]).transpose(1,2,0)
    for l in label_infos:
        mask = label_img == l['id']
        label_img_rgb[mask] = l['rgb_values']

    return label_img_rgb.astype(np.uint8)

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

    def get_item_from_index(self, index):
        
        to_tensor = transforms.ToTensor()
        img_id = self.image_names[index].replace('.png','')

        print("img_id:", img_id)

        img = Image.open(os.path.join(self.root_dir_name,
                                      'raw',
                                      'right',
                                      img_id + '.png')).convert('RGB')

        img = to_tensor(img)

        target = Image.open(os.path.join(self.root_dir_name,
                                      'labeled',
                                      'right',
                                      img_id + '.png'))

        target = np.array(target, dtype=np.int64)

        target_labels = target[..., 0]
        for label in LABELS_LIST:
            mask = np.all(target == label['rgb_values'], axis = 2)
            target_labels[mask] = label['id']

        target_labels = torch.from_numpy(target_labels.copy())

        return img, target_labels