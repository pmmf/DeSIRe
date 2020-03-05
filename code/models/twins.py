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
from cvae import CVAE
from cnn import CNN


class TWINS(nn.Module):
    def __init__(self,
                 input_shape=(3, 64, 64),
                 n_labels=10,
                 n_ids=10,
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

        super(TWINS, self).__init__()

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

        # Initialize main CVAE
        self.cvae = CVAE(input_shape=self.input_shape,
                         n_labels=self.n_labels,
                         n_ids=self.n_ids,
                         base_filters=self.base_filters,
                         kernel_size=self.kernel_size,
                         stride=self.stride,
                         learning_rate=self.learning_rate,
                         l2_reg=self.l2_reg,
                         activation=self.activation,
                         zdims=self.zdims,
                         bnorm=self.bnorm,
                         batch_size=self.batch_size,
                         KLD_weight=self.KLD_weight,
                         DeconvIsConv=self.DeconvIsConv)

        # Initialize CNN
        self.cnn = CNN(input_shape=self.input_shape,
                       isTWINS=True)

    def forward(self, x, y, y_1D, g_decoder_1D, y_2D, g_2D,
                x2, g2_2D):

        # cvae
        x_reconst, z_mean, z_log_var, z = self.cvae(x, y_1D, g_decoder_1D,
                                                    y_2D, g_2D)

        # shared cvae encoder
        z_mean2, z_log_var2 = self.cvae.encoder(x2, y_2D, g2_2D)

        # cnn
        h1, y_pred = self.cnn(x)

        return (x_reconst, z_mean, z_log_var, z,
                z_mean2, z_log_var2,
                h1, y_pred)

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
    model = CNN(input_shape=input_shape).to(device)

    print(model)
    summary(model, input_shape)
    # print(model.flattenDims)
