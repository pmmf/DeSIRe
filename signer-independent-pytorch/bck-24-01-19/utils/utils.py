import sys
import math
sys.path.insert(0, './layers/')
from layers import get_padding
import numpy as np

import torch


def annealing_function(init_weight, ftype='exponential', step=0, k=0.0025, x0=2500):
    if ftype == 'exponential':
        return init_weight * float(1 / (1 + np.exp(-k*(step-x0))))
    elif ftype == 'linear':
        return init_weight * min(1, step/x0)
    else:
        raise Warning("Invalid annealing function!")


def one_hot_1D(n_classes, label):
    one_hot = torch.zeros(n_classes,)
    one_hot[label] = 1
    return one_hot


def one_hot_2D(n_classes, size, label):
    one_hot = torch.zeros(n_classes, size[0], size[1])
    one_hot[label, :, :] = 1
    return one_hot


def merge_images(images, size):
    # merge all output images(of sample size:8*8 output images of size 64*64)
    # into one big image
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):  # idx=0,1,2,...,63
        i = idx % size[1]  # column number
        j = idx // size[1]  # row number
        img[j*h:j*h+h, i*w:i*w+w, :] = image
    return img


def inverse_transform(x, isInGPU=True):
    if isInGPU:
        x = x.to('cpu').numpy()
    else:
        x = x.numpy()

    if len(x.shape) == 4:
        x = x.transpose((0, 2, 3, 1))
    elif len(x.shape) == 3:
        x = x.transpose((1, 2, 0))

    x = (x+1.)/2.
    return x


def get_deconvblock_padding(input_dim, kernel_sizes, strides, out_dims):
    ''' Get size of each deconv layer output '''
    H_in, W_in = input_dim
    out_H, out_W = out_dims
    out_pad = []

    for i in range(len(kernel_sizes)):
        # H_out
        H = int((H_in-1)*strides[i] - 2*get_padding(kernel_sizes[i]) +
                kernel_sizes[i])
        # out_padding = H - H_desired
        out_h = out_H[i]-H
        # update H_in
        H_in = H + out_h

        # W_out
        W = int((W_in-1)*strides[i] - 2*get_padding(kernel_sizes[i]) +
                kernel_sizes[i])
        # out_padding = W - W_desired
        out_w = out_W[i]-W
        # update W_in
        W_in = W + out_w

        # Append output paddings
        out_pad.append((out_h, out_w))

    return out_pad


def get_convblock_dim(input_dim, conv_filters, kernel_sizes, strides):
    ''' Get size of each conv layer output '''
    _, H, W = input_dim
    h_list = [H]
    w_list = [W]
    for i in range(len(conv_filters)):
        H = int(math.floor((H + 2*get_padding(kernel_sizes[i]) -
                kernel_sizes[i])/(1.*strides[i]) + 1))
        h_list.append(H)

        W = int(math.floor((W + 2*get_padding(kernel_sizes[i]) -
                kernel_sizes[i])/(1.*strides[i]) + 1))
        w_list.append(W)

    return h_list, w_list


def get_flat_dim(input_dim, conv_filters, kernel_sizes, strides):
    ''' Get flatten dimension after conv block '''
    _, H, W = input_dim
    for i in range(len(conv_filters)):
        H = int(math.floor((H + 2*get_padding(kernel_sizes[i]) -
                kernel_sizes[i])/(1.*strides[i]) + 1))
        W = int(math.floor((W + 2*get_padding(kernel_sizes[i]) -
                kernel_sizes[i])/(1.*strides[i]) + 1))

    flat_dim = H * W * conv_filters[-1]

    return flat_dim

if __name__ == '__main__':
    pass