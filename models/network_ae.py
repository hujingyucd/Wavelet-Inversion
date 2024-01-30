import os
import time
import math
import random
import numpy as np
import h5py

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from models.network import NearestUpsample3D, Conv3D


class VoxelEncoder(nn.Module):
    def __init__(self, config):
        super(VoxelEncoder, self).__init__()
        self.ef_dim = config.ae_ef_dim
        self.z_dim = config.ae_z_dim
        self.ae_input_channel = config.ae_input_channel
        self.slope = config.negative_slope
        self.conv_1 = nn.Conv3d(self.ae_input_channel, self.ef_dim, 4, stride=2, padding=1, bias=False)
        self.in_1 = nn.InstanceNorm3d(self.ef_dim)
        self.conv_2 = nn.Conv3d(self.ef_dim, self.ef_dim * 2, 4, stride=2, padding=1, bias=False)
        self.in_2 = nn.InstanceNorm3d(self.ef_dim * 2)
        self.conv_3 = nn.Conv3d(self.ef_dim * 2, self.ef_dim * 4, 4, stride=2, padding=1, bias=False)
        self.in_3 = nn.InstanceNorm3d(self.ef_dim * 4)
        self.conv_4 = nn.Conv3d(self.ef_dim * 4, self.ef_dim * 8, 4, stride=2, padding=1, bias=False)
        self.in_4 = nn.InstanceNorm3d(self.ef_dim * 8)
        self.conv_5 = nn.Conv3d(self.ef_dim * 8, self.z_dim, 4, stride=1, padding=0, bias=True)
        nn.init.xavier_uniform_(self.conv_1.weight)
        nn.init.xavier_uniform_(self.conv_2.weight)
        nn.init.xavier_uniform_(self.conv_3.weight)
        nn.init.xavier_uniform_(self.conv_4.weight)
        nn.init.xavier_uniform_(self.conv_5.weight)
        nn.init.constant_(self.conv_5.bias, 0)

    def forward(self, inputs):
        d_1 = self.in_1(self.conv_1(inputs))
        d_1 = F.leaky_relu(d_1, negative_slope=self.slope, inplace=True)

        d_2 = self.in_2(self.conv_2(d_1))
        d_2 = F.leaky_relu(d_2, negative_slope=self.slope, inplace=True)

        d_3 = self.in_3(self.conv_3(d_2))
        d_3 = F.leaky_relu(d_3, negative_slope=self.slope, inplace=True)

        d_4 = self.in_4(self.conv_4(d_3))
        d_4 = F.leaky_relu(d_4, negative_slope=self.slope, inplace=True)

        d_5 = self.conv_5(d_4)
        # d_5 = d_5.view(-1, self.z_dim)
        # d_5 = torch.sigmoid(d_5)

        return d_5

class VoxelDecoder(nn.Module):
    def __init__(self, config):
        super(VoxelDecoder, self).__init__()
        # ef_dim = config.ae_ef_dim
        # z_dim = config.ae_z_dim
        self.ae_input_channel = config.ae_input_channel
        self.z_dim = config.ae_z_dim
        self.ef_dim = config.ae_ef_dim
        self.slope = config.negative_slope
        self.conv_1 = nn.ConvTranspose3d(self.z_dim, self.ef_dim * 8, 4, stride=1, padding=0, bias=True)
        self.in_1 = nn.InstanceNorm3d(self.ef_dim)
        self.conv_2 = nn.ConvTranspose3d(self.ef_dim * 8, self.ef_dim * 4, 4, stride=2, padding=1, bias=False)
        self.in_2 = nn.InstanceNorm3d(self.ef_dim * 2)
        self.conv_3 = nn.ConvTranspose3d(self.ef_dim * 4, self.ef_dim * 2, 4, stride=2, padding=1, bias=False)
        self.in_3 = nn.InstanceNorm3d(self.ef_dim * 4)
        self.conv_4 = nn.ConvTranspose3d(self.ef_dim * 2, self.ef_dim * 1, 4, stride=2, padding=1, bias=False)
        self.in_4 = nn.InstanceNorm3d(self.ef_dim * 8)
        self.conv_5 = nn.ConvTranspose3d(self.ef_dim , self.ae_input_channel, 4, stride=2, padding=1, bias=False)
        nn.init.xavier_uniform_(self.conv_1.weight)
        nn.init.xavier_uniform_(self.conv_2.weight)
        nn.init.xavier_uniform_(self.conv_3.weight)
        nn.init.xavier_uniform_(self.conv_4.weight)
        nn.init.xavier_uniform_(self.conv_5.weight)
        nn.init.constant_(self.conv_1.bias, 0)

    def forward(self, inputs):
        d_1 = self.in_1(self.conv_1(inputs))
        d_1 = F.leaky_relu(d_1, negative_slope=self.slope, inplace=True)

        d_2 = self.in_2(self.conv_2(d_1))
        d_2 = F.leaky_relu(d_2, negative_slope=self.slope, inplace=True)

        d_3 = self.in_3(self.conv_3(d_2))
        d_3 = F.leaky_relu(d_3, negative_slope=self.slope, inplace=True)

        d_4 = self.in_4(self.conv_4(d_3))
        d_4 = F.leaky_relu(d_4, negative_slope=self.slope, inplace=True)

        d_5 = self.conv_5(d_4)
        # d_5 = d_5.view(-1, self.z_dim)
        # d_5 = torch.sigmoid(d_5)

        return d_5


class VoxelAutoEncoder(nn.Module):
    def __init__(self, config):
        super(VoxelAutoEncoder, self).__init__()
        self.config = config
        self.encoder = VoxelEncoder(self.config)
        if hasattr(self.config, "use_trans3d") and self.config.use_trans3d:
            print("Decoder use Transposed3D")
            self.decoder = VoxelDecoder(self.config)
        else:
            print("Decoder use upsampling")
            self.decoder = Conv3D(self.config.latent_dim, self.config.ae_input_channel, config)

    def forward(self, inputs, latent_codes = None):
        if latent_codes is None:
            latent_codes = self.encoder(inputs)
        if hasattr(self.config, "use_trans3d") and self.config.use_trans3d:
            pred_voxels = self.decoder(latent_codes)
        else:
            pred_voxels = self.decoder(latent_codes.view(self.config.batch_size, -1), [32, 32, 32])

        return pred_voxels, latent_codes

    @classmethod
    def recon_loss(self, inputs, preds):
        diff = inputs - preds
        recon_loss = (diff**2).mean()
        return recon_loss