import sys
import math
sys.path.insert(0, './layers/')
from layers import get_padding
import numpy as np

import torch

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import copy


def tsne(model, data_loader, device, plot_fn=None):
    markers = ['.', 'v', '1', 'p', 'P', '*', 'X', '8', '+', 'x', 'd', '|', 's', '>']
    colors = ['black', 'red', 'blue', 'green', 'orange', 'fuchsia', 'lime', 'peru', 'cyan', 'rosybrown']

    from sklearn import manifold
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    # X_tsne = tsne.fit_transform(X)

    with torch.no_grad():
        # set model to test
        model.eval()

        x_array = []
        y_array = []
        g_array = []
        for i, *atuple in enumerate(data_loader):

            if hasattr(model, 'encoder'):
                x = atuple[0][0].to(device)
                y = atuple[0][1].to(device)
                g = atuple[0][2].to(device)
                y_2D = atuple[0][5].to(device)
                g_2D = atuple[0][6].to(device)

                # forward pass
                h1, _ = model.encoder(x, y_2D, g_2D)
            else:
                # send mini-batch to gpu
                x = atuple[0][0].to(device)
                y = atuple[0][1].to(device)
                g = atuple[0][2].to(device)

                # forward pass (INFERENCE: JUST ON CLASSIFIER)
                h1, _ = model(x)

            # concat data
            if i == 0:
                x_array = copy.deepcopy(h1)
                y_array = copy.deepcopy(y)
                g_array = copy.deepcopy(g)
                continue

            x_array = torch.cat((x_array, h1), dim=0)
            y_array = torch.cat((y_array, y), dim=0)
            g_array = torch.cat((g_array, g), dim=0)

    x_array = x_array.to('cpu').numpy()
    y_array = y_array.to('cpu').numpy()
    g_array = g_array.to('cpu').numpy()

    X_tsne = tsne.fit_transform(x_array)

    plt.figure()
    for i in range(X_tsne.shape[0]):
        # markers
        plt.scatter(X_tsne[i, 0], X_tsne[i, 1], color=colors[y_array[i]], marker=markers[g_array[i]])
    if plot_fn:
        plt.show()
        plt.savefig(plot_fn)
        plt.close()
    else:
        plt.show()


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