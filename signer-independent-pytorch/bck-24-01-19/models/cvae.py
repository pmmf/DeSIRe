import sys
sys.path.insert(0, './data/')
sys.path.insert(0, './layers/')
sys.path.insert(0, './utils/')

import math

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import ShuffleSplit
from torch import Tensor

from data import CelebA
from layers import (BasicConvLayer, BasicDeconvLayer, BasicDenseLayer,
                    get_padding)
from torchsummary import summary
from utils import (get_flat_dim, get_convblock_dim, get_deconvblock_padding)


class ENCODER(nn.Module):
    def __init__(self,
                 input_shape=(3, 64, 64),
                 n_attributes=10,
                 base_filters=[64, 128, 256, 512],
                 kernel_size=[3, 3, 3, 3],
                 stride=[2, 2, 2, 2],
                 activation='leaky_relu',
                 zdims=128,
                 bnorm=True):

        super(ENCODER, self).__init__()

        self.input_shape = input_shape
        self.n_attributes = n_attributes
        self.base_filters = base_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.activation = activation
        self.zdims = zdims
        self.bnorm = bnorm

        # Initialize encoder layers
        self.create_encoder_layers()

    def create_encoder_layers(self):
        # Conv layers
        # first conv layer
        conv_list = nn.ModuleList([
            BasicConvLayer(in_channels=self.input_shape[0] + self.n_attributes,
                           out_channels=self.base_filters[0],
                           bnorm=self.bnorm,
                           activation=self.activation,
                           kernel_size=self.kernel_size[0],
                           stride=self.stride[0])
        ])

        # remaining conv layers
        conv_list.extend([BasicConvLayer(in_channels=self.base_filters[l-1],
                                         out_channels=self.base_filters[l],
                                         bnorm=self.bnorm,
                                         activation=self.activation,
                                         kernel_size=self.kernel_size[l],
                                         stride=self.stride[l])
                         for l in range(1, len(self.base_filters))])

        # ConvBlock
        self.ConvBlock = nn.Sequential(*conv_list)

        # Dense layers
        flat_dim = get_flat_dim(self.input_shape, self.base_filters,
                                self.kernel_size, self.stride)
        # print(flat_dim)

        # Mean
        self.z_mean = BasicDenseLayer(in_features=flat_dim,
                                      out_features=self.zdims,
                                      bnorm=self.bnorm,
                                      activation='linear')

        # Mean
        self.z_log_var = BasicDenseLayer(in_features=flat_dim,
                                         out_features=self.zdims,
                                         bnorm=self.bnorm,
                                         activation='linear')

    def forward(self, x, y_2D, g_2D):
        # conv blocks
        inputs = torch.cat((x, y_2D, g_2D), 1)
        x = self.ConvBlock(inputs)

        # flatten
        x = x.reshape(x.size(0), -1)

        # z_mean
        z_mean = self.z_mean(x)

        # z_log_var
        z_log_var = self.z_log_var(x)

        return z_mean, z_log_var


class DECODER(nn.Module):
    def __init__(self,
                 input_shape=(3, 64, 64),
                 n_attributes=10,
                 base_filters=[64, 128, 256, 512],
                 kernel_size=[3, 3, 3, 3],
                 stride=[2, 2, 2, 2],
                 activation='leaky_relu',
                 zdims=1,
                 bnorm=True,
                 DeconvIsConv=True):

        super(DECODER, self).__init__()

        self.input_shape = input_shape
        self.n_attributes = n_attributes
        self.base_filters = base_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.activation = activation
        self.zdims = zdims
        self.bnorm = bnorm
        self.DeconvIsConv = DeconvIsConv

        # Initialize decoder layers
        self.create_decoder_layers()

    def create_decoder_layers(self):
        # feature maps size and number of filters of the first layer
        hConvOut, wConvOut = get_convblock_dim(self.input_shape,
                                               self.base_filters,
                                               self.kernel_size, self.stride)

        self.feat_sze_h, self.feat_sze_w = hConvOut[-1], wConvOut[-1]
        hConvOut.pop(-1)
        wConvOut.pop(-1)

        # if denconv is not transpose of conv
        deconv_index = -1
        if not self.DeconvIsConv:
            deconv_index -= 1

        self.n_filt = self.base_filters[deconv_index]

        # first dense
        out_features = self.feat_sze_h*self.feat_sze_w*self.n_filt
        self.dense = BasicDenseLayer(in_features=self.zdims + self.n_attributes,
                                     out_features=out_features,
                                     activation=self.activation,
                                     bnorm=self.bnorm)
        # Deconv layers
        # reserve net architecture lists
        income_channels = self.base_filters[deconv_index::-1]
        income_channels.insert(0, self.n_filt)
        income_channels.pop(-1)
        outcome_channels = self.base_filters[deconv_index::-1]
        reverse_kernel = self.kernel_size[deconv_index::-1]
        reverse_stride = self.stride[deconv_index::-1]
        hConvOut = hConvOut[deconv_index::-1]
        wConvOut = wConvOut[deconv_index::-1]

        # get output paddings
        out_pad = get_deconvblock_padding((self.feat_sze_h, self.feat_sze_w),
                                          reverse_kernel, reverse_stride,
                                          (hConvOut, wConvOut))

        # deconv layers
        deconv_list = nn.ModuleList([
            BasicDeconvLayer(in_channels=income_channels[l],
                             out_channels=outcome_channels[l],
                             activation=self.activation,
                             bnorm=self.bnorm,
                             output_padding=out_pad[l][0],
                             kernel_size=reverse_kernel[l],
                             stride=reverse_stride[l])
            for l in range(len(income_channels))])

        # output layer - NO BN!
        deconv_list.extend([BasicDeconvLayer(in_channels=outcome_channels[-1],
                                             out_channels=self.input_shape[0],
                                             kernel_size=self.kernel_size[0],
                                             stride=1,
                                             output_padding=0,
                                             bnorm=False,
                                             activation='tanh')])

        # DeconvBlock
        self.DenconvBlock = nn.Sequential(*deconv_list)

    def forward(self, z, y_1D, g_1D):
        # y_1D: passed bu tnot used
        inputs = torch.cat((z, g_1D), 1)
        h = self.dense(inputs)
        h = h.view(-1, self.n_filt, self.feat_sze_h, self.feat_sze_w)
        h = self.DenconvBlock(h)
        return h


class CVAE(nn.Module):
    def __init__(self,
                 input_shape=(3, 64, 64),
                 n_labels=10,
                 n_ids=14,
                 base_filters=[64, 64, 128, 256, 512],
                 kernel_size=[3, 3, 3, 3, 3],
                 stride=[2, 2, 2, 2, 2],
                 learning_rate=1e-03,
                 l2_reg=0,
                 activation='leaky_relu',
                 zdims=128,
                 bnorm=True,
                 batch_size=32,
                 KLD_weight=5e-04,
                 DeconvIsConv=True,
                 checkpoint_fn='./checkpoints/'):  # 5e-04

        super(CVAE, self).__init__()

        self.input_shape = input_shape
        self.n_labels = n_labels
        self.n_ids = n_ids
        self.base_filters = base_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.l2_reg = l2_reg
        self.activation = activation
        self.zdims = zdims
        self.bnorm = bnorm
        self.batch_size = batch_size
        self.KLD_weight = KLD_weight
        self.learning_rate = learning_rate
        self.DeconvIsConv = DeconvIsConv

        # Initialize Encoder
        self.encoder = ENCODER(input_shape=self.input_shape,
                               base_filters=self.base_filters,
                               kernel_size=self.kernel_size,
                               stride=self.stride,
                               activation=self.activation,
                               zdims=self.zdims,
                               bnorm=self.bnorm,
                               n_attributes=self.n_labels + self.n_ids)

        # Initialize Decoder
        self.decoder = DECODER(input_shape=self.input_shape,
                               base_filters=self.base_filters,
                               kernel_size=self.kernel_size,
                               stride=self.stride,
                               activation=self.activation,
                               zdims=self.zdims,
                               bnorm=self.bnorm,
                               DeconvIsConv=self.DeconvIsConv,
                               n_attributes=self.n_ids)

    def reparameterize(self, z_mean, z_log_var):
        eps = torch.randn_like(z_mean)
        z = z_mean + torch.exp(.5*z_log_var) * eps

        return z

    def forward(self, x, y_1D, g_1D, y_2D, g_2D):
        # encode
        z_mean, z_log_var = self.encoder(x, y_2D, g_2D)

        # reparameterization trick
        z = self.reparameterize(z_mean, z_log_var)

        # decode
        x_reconst = self.decoder(z, y_1D, g_1D)  # y_1D: passed but not used
        return x_reconst, z_mean, z_log_var, z

    def fit(self):
        pass

    def new_samples(self):
        pass

    def encode_decode(self):
        pass


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    input_shape = (3, 64, 64)
    # input_shape = (1024,)
    model = CVAE().to(device)

    print(model)
    summary(model, input_shape)
    # print(model.flattenDims)
