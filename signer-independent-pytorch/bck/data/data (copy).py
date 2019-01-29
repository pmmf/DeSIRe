import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from skimage import io
from sklearn.model_selection import (ShuffleSplit, StratifiedShuffleSplit,
                                     GroupShuffleSplit)
from torch.utils.data import Dataset
from torchvision import transforms

import random
import copy

from utils.utils import one_hot_1D, one_hot_2D


def inverse_transform(x):
    x = x.to('cpu').numpy()

    x = x.transpose((1, 2, 0))
    x = (x+1.)/2.
    return x


def getTrainTestIndexes(method='groups'):
    pass


def split_data(dataset, batch_sze, stratify=False, groups=False,
               test_size=.15):
    # Split train and validation
    X_to_split = np.zeros((len(dataset), 1))
    if groups:
        rs = GroupShuffleSplit(n_splits=1, test_size=test_size,
                               random_state=42)
        g = [dataset[i][2] for i in range(len(dataset))]
        tr_indexes, test_indexes = next(rs.split(X_to_split, groups=g))
    elif stratify:
        rs = StratifiedShuffleSplit(n_splits=1, test_size=test_size,
                                    random_state=42)
        y = [dataset[i][1] for i in range(len(dataset))]
        tr_indexes, test_indexes = next(rs.split(X_to_split, y))
    else:
        rs = ShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
        tr_indexes, test_indexes = next(rs.split(X_to_split))

    # train and validation loaders
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(tr_indexes)
    valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(test_indexes)

    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=batch_sze,
                                               shuffle=False, num_workers=4,
                                               sampler=train_sampler)

    valid_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=batch_sze,
                                               shuffle=False, num_workers=4,
                                               sampler=valid_sampler)

    return train_loader, valid_loader


class Normalize(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        sample = sample*2 - 1.
        return sample


class CelebA(Dataset):
    def __init__(self,
                 data_fn='/data/DB/celebA/img_align_celeba/',
                 transform=True):

        self.transform = transform
        self.data_fn = data_fn
        self.imgs_list = sorted(os.listdir(self.data_fn))

    def __len__(self):
        return len(self.imgs_list)

    def __getitem__(self, index):
        img = Image.open(os.path.join(*(self.data_fn, self.imgs_list[index])))

        if self.transform:
            self.tranforms = self.tranformations()
            img = self.tranforms(img)

        return img

    def tranformations(self):
        # celeba transforms
        crop_size = (148, 148)
        resize_sze = (64, 64)
        data_transform = transforms.Compose([
                transforms.CenterCrop(crop_size),
                transforms.Resize(resize_sze),
                transforms.ToTensor(),
                Normalize()])
        return data_transform


class KinectLeap(Dataset):
    def __init__(self,
                 data_fn='/data/DB/kinect_leap_dataset_signer_independent/',
                 n_person=14,
                 n_gesture=10,
                 n_repetions=10,
                 extension='_rgb.png',
                 data_type='RGB_CROPS_RZE_DISTTR2',
                 as_gray=False,
                 validation=None,
                 transform=True,
                 train_groups=[1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 13],
                 model='cnn'):  # 'cnn', 'vae', 'twins'

        self.data_fn = data_fn
        self.n_person = n_person
        self.n_gesture = n_gesture
        self.n_repetions = n_repetions
        self.data_type = data_type
        self.extension = extension
        self.as_gray = as_gray
        self.validation = validation
        self.transform = transform
        self.model = model
        self.train_groups = train_groups

        self.imgs_list = sorted(os.listdir(self.data_fn))

    def set_train_groups(self, train_groups):
        self.train_groups = train_groups

    def __len__(self):
        return self.n_person*self.n_gesture*self.n_repetions

    def index2dataset(self, index):
        # get person, gesture and repetion indexes from global index
        n_rep = index % self.n_repetions + 1
        n_gesture = int(np.floor(index/self.n_gesture) % self.n_repetions) + 1
        n_person = int(np.floor(index/(self.n_gesture*self.n_repetions))) + 1

        return n_person, n_gesture, n_rep

    def __getitem__(self, index):
        # get indexes
        n_person, n_gesture, n_rep = self.index2dataset(index)
        y = n_gesture - 1
        g = n_person - 1

        # read image
        img_fn = os.path.join(*[self.data_fn,
                                "P" + str(n_person),
                                "G" + str(n_gesture),
                                self.data_type,
                                str(n_rep) + self.extension])
        # print(index, img_fn)
        x = Image.open(img_fn)
        h, w = x.size

        # 1D one hot encoding
        # labels
        y_1D = one_hot_1D(n_classes=self.n_gesture, label=y)
        # person id
        g_1D = one_hot_1D(n_classes=self.n_person, label=g)

        # 2D one hot encoding
        # labels
        y_2D = one_hot_2D(n_classes=self.n_gesture, size=(h, w),
                                label=y)

        # person id
        g_2D = one_hot_2D(n_classes=self.n_person, size=(h, w),
                                label=g)

        # transform
        if self.transform:
            self.tranforms = self.tranformations()
            x = self.tranforms(x)

        if self.model == 'cnn':
            return x, y, g
        elif self.model == 'vae':
            return x, y  # , n_person-1
        elif self.model == 'cvae':
            return x, y, g, y_1D, g_1D, y_2D, g_2D
        elif self.model == 'twins':
            # INPUTS FOR PRINCIPAL CVAE DECODER
            # there is 50% of changing id
            p = 0.5
            is2change = np.random.choice([0,1], 1, p=[1-p, p])

            g_decoder = copy.deepcopy(g)
            g_decoder_1D = copy.deepcopy(g_1D)
            x_decoder = copy.deepcopy(x)
            if is2change:
                while g_decoder == g:
                    # g_decoder = random.randint(0, self.n_person-1)
                    g_decoder = random.randint(0, len(self.train_groups)-1)
                    g_decoder = self.train_groups[g_decoder]
                g_decoder_1D = one_hot_1D(n_classes=self.n_person, label=g_decoder)
                img_fn = os.path.join(*[self.data_fn,
                        "P" + str(g_decoder + 1),
                        "G" + str(n_gesture),
                        self.data_type,
                        str(n_rep) + self.extension])

                x_decoder = Image.open(img_fn)
                # transform
                if self.transform:
                    self.tranforms = self.tranformations()
                    x_decoder = self.tranforms(x_decoder)

            # INPUTS FOR EXTRA CVAE ENCODER
            # generate person id g2 (g2 != g)
            g2 = copy.deepcopy(g)
            g2_2D = copy.deepcopy(g_2D)
            while g2 == g:
                # g2 = random.randint(0, self.n_person-1)
                g2 = random.randint(0, len(self.train_groups)-1)
                g2 = self.train_groups[g2]

            g2_2D = one_hot_2D(n_classes=self.n_person, size=(h, w),
                                  label=g2)

            img_fn = os.path.join(*[self.data_fn,
                        "P" + str(g2 + 1),
                        "G" + str(n_gesture),
                        self.data_type,
                        str(n_rep) + self.extension])

            x2 = Image.open(img_fn)
            # transform
            if self.transform:
                self.tranforms = self.tranformations()
                x2 = self.tranforms(x2)

            return(x, y, g, y_1D, g_1D, y_2D, g_2D,
                   x_decoder, g_decoder, g_decoder_1D,
                   x2, g2, g2_2D)


    def tranformations(self):
        # celeba transforms
        data_transform = transforms.Compose([transforms.ToTensor(),
                                            Normalize()])
        return data_transform


if __name__ == '__main__':
    crop_size = 148
    resize_sze = (64, 64)
    BATCH_SIZE = 32

    dataset = KinectLeap(model='cvae')

    for i in range(len(dataset)):
        X, y, g, y_, y__, g_, g__ = dataset[i]
        X = inverse_transform(X)
        print(y)
        print(g)
        print(y_)
        print(g_)
        print(y__)
        print(g__)
        plt.figure()
        plt.imshow(X)
        plt.axis('off')
        plt.show()


    # Data
    # data transform
    data_transform = transforms.Compose([
                     transforms.CenterCrop((crop_size, crop_size)),
                     transforms.Resize(resize_sze),
                     transforms.ToTensor(),
                     Normalize()])

    dataset = CelebA(data_fn='/data/DB/celebA/img_align_celeba/',
                     transform=data_transform)

    dataset_len = len(dataset)

    # Split train and validation
    rs = ShuffleSplit(n_splits=1, test_size=.25, random_state=42)
    X_to_split = np.zeros((dataset_len, 1))

    tr_indexes, test_indexes = next(rs.split(X_to_split))

    # train and validation loaders
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(tr_indexes)
    valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(test_indexes)

    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=BATCH_SIZE,
                                               shuffle=False, num_workers=4,
                                               sampler=train_sampler)

    valid_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=BATCH_SIZE,
                                               shuffle=False, num_workers=4,
                                               sampler=valid_sampler)

    print(dataset[0].shape)

    for i in range(len(dataset)):
        X = inverse_transform(dataset[i])
        # print(y, group)
        plt.figure()
        plt.imshow(X)
        plt.axis('off')
        plt.show()
