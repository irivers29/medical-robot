"""IMPORTS"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision.models as models
from torch.utils.data import random_split, DataLoader


class SegmentationAE(pl.LightningModule):

    def __init__(self, num_classes=5, hparams=None):
        super().__init__()
        self.hparams = hparams
        
        """ define the blocks for the encoder """

        self.down_conv1 = self._UNet_block(3, 64)
        self.down_conv2 = self._UNet_block(64, 128)
        self.down_conv3 = self._UNet_block(128, 256)
        self.down_conv4 = self._UNet_block(256, 512)

        """ define the blocks for the decoder """

        self.up_conv1 = self._UNet_block(512 + 256, 256)
        self.up_conv2 = self._UNet_block(256 + 128, 128)
        self.up_conv3 = self._UNet_block(128 + 64, 64)
        #self.up_conv3 = self._UNet_block(64 + 5, 5)

        """ define 1x1 convolution for last block """

        self.one_by_one = nn.Conv2d(64, num_classes, kernel_size = 1)

        """ define 2x2 max_pool and 2x2 up-conv """

        self.maxpool = nn.MaxPool2d( kernel_size = 2)
        self.upsampling = nn.Upsample(scale_factor=2)

    def forward(self, x):

        """ use blocks for creating the U-Net architecture """

        first_block = self.down_conv1(x)
        x = self.max_pool(first_block)

        second_block = self.down_conv2(x)
        x = self.max_pool(second_block)

        third_block = self.down_conv3(x)
        x = self.max_pool(third_block)

        fourth_block = self.down_conv4(x)

        x = self.upsampling(fourth_block)
        x = torch.cat([x, third_block], dim = 1)

        x = self.up_conv1(x)
        x = self.upsampling(x)
        x = torch.cat([x, second_block], dim = 1)

        x = self.up_conv2(x)
        x = self.upsampling(x)
        x = torch.cat([x, first_block], dim = 1)

        x = self.up_conv1(x)

        x = self.one_by_one(x)

        return x


    @staticmethod
    def _UNet_block(in_channels, out_channels, kernel_size = 3):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding = 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding = 1),
            nn.ReLU(inplace=True)
        )

        return block

    def training_step(self, train_batch, train_batch_idx):
        
        images, targets = train_batch
        criterion = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction = 'mean')

        """ do forward step """
        forward_output = self.forward(images)

        loss = criterion(forward_output, targets)

        return {'loss': loss}
        
    def train_dataloader(self, train_data):
        return DataLoader(train_data)

    def val_dataloader(self, val_data):
        return DataLoader(val_data)

    def test_dataloader(self, test_data):
        return DataLoader(test_data)

    def validation_step(self, val_batch, batch_idx):

        images, targets = val_batch
        criterion = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')

        """ do forward step """
        forward_output = self.forward(images)

        val_loss = criterion(forward_output, targets)

        return {'val_loss': val_loss}

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
