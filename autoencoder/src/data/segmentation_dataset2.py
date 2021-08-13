"""Data utility functions."""
import os
from posix import POSIX_FADV_WILLNEED
import cv2
import numpy as np
from google.colab.patches import cv2_imshow
import torch
import torch.utils.data as data
from PIL import Image
from torchvision import transforms
import os

import _pickle as pickle


LABELS_LIST = [
    {"id": 0, "name":"void", "rgb_values": [0, 0, 0]},
    {"id": 1, "name":"PIP", "rgb_values": [250, 50, 183]},
    {"id": 2, "name":"MCP", "rgb_values": [236, 28, 36]},
    {"id": 3, "name":"CMC", "rgb_values": [14, 209, 69]},
    {"id": 4, "name":"Wrist", "rgb_values": [255,242,0]}
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


        #img = Image.open(os.path.join("/content/drive/MyDrive/summer_school_sdu/ROPCA_Vision/raw/all/" +
        #                              img_id + '.png'))
        
        img = Image.open(os.path.join("/content/drive/MyDrive/summer_school_sdu/ROPCA_Vision/raw/left/" +
                                      img_id + '.png'))
        
        crop_zone = (121,300,505,732)
        cropped_img = img.crop(crop_zone)
        width,height = cropped_img.size
        new_width = (int)(width/2)
        new_height = (int)(height/2)
        resized_img = cropped_img.resize((new_width,new_height))

        img = to_tensor(cropped_img)
        
        #target = Image.open(os.path.join("/content/drive/MyDrive/summer_school_sdu/ROPCA_Vision/labeled/all/" +
        #                              img_id + '.png'))
        target = Image.open(os.path.join("/content/drive/MyDrive/summer_school_sdu/ROPCA_Vision/labeled/left/" +
                                      img_id + '.png'))

        #cropped__target_img = target.crop(crop_zone)
        #width,height = cropped_img.size
        #new_width = (int)(width/2)
        #new_height = (int)(height/2)
        #resized_target_img = cropped_img.resize((new_width,new_height))

        cropped_target = target.crop(crop_zone)
        width,height = cropped_target.size
        new_width = (int)(width/2)
        new_height = (int)(height/2)
        resized_target_img = cropped_img.resize((new_width,new_height))

        target = np.array(cropped_target, dtype=np.int64)
        values = [np.array([0,0,0])]

        target_labels = target[..., 0]
        for label in LABELS_LIST:
            mask = np.all(target == label['rgb_values'], axis = 2)
            target_labels[mask] = label['id']

        target_labels = torch.from_numpy(target_labels.copy())

        return img, target_labels