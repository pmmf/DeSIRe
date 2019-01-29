import os
import sys

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
from random import randint
import copy

sys.path.insert(0, './utils/')
from utils import one_hot_1D, one_hot_2D, inverse_transform
sys.path.insert(0, './data/')
from transforms import (numpyToTensor, Normalize_1_1, Normalize_0_1,
                        randomHorizontalFlip, randomAffineTransform,
                        randomColourTransform, ImgAugTransform)

# DATASETS_LIST = {'celeba': CelebA, 'staticSL': KinectLeap}


def getSplitter(dataset, n_splits=5, mode='groups', test_size=.15):
    X_to_split = np.zeros((len(dataset), 1))
    if mode == 'groups':
        rs = GroupShuffleSplit(n_splits=n_splits, test_size=test_size,
                               random_state=42)
        g = [dataset[i][2] for i in range(len(dataset))]
        # tr_indexes, test_indexes = next(splitter.split(X_to_split, groups=g))
        return rs.split(X_to_split, groups=g)
    elif mode == 'stratify':
        rs = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size,
                                    random_state=42)

        y = [dataset[i][1] for i in range(len(dataset))]
        return rs.split(splitter.split(X_to_split, y))
    else:
        rs = ShuffleSplit(n_splits=n_splits, test_size=test_size,
                          random_state=42)
        return rs.split(X_to_split)


def split_data(dataset, splitter, batch_sze, dataAug, mode='groups'):
    # Split train and validation
    X_to_split = np.zeros((len(dataset), 1))

    tr_indexes, test_indexes = splitter
    g_train = list(np.unique([dataset[i][2] for i in tr_indexes]))
    g_test = list(np.unique([dataset[i][2] for i in test_indexes]))

    print(g_train, g_test)

    # train and validation loaders
    data_train = KinectLeap(model=dataset.model, isTrain=True,
                            person_ids=g_train, dataAug=dataAug)
    data_valid = KinectLeap(model=dataset.model, isTrain=False,
                            person_ids=g_test, dataAug=False)

    train_loader = torch.utils.data.DataLoader(data_train,
                                               batch_size=batch_sze,
                                               shuffle=True, num_workers=4,
                                               sampler=None)

    valid_loader = torch.utils.data.DataLoader(data_valid,
                                               batch_size=batch_sze,
                                               shuffle=False, num_workers=4,
                                               sampler=None)

    return train_loader, valid_loader


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
                 isTrain=False,
                 transform=True,
                 dataAug=False,
                 person_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                 model='cnn'):  # 'cnn', 'vae', 'twins'

        self.data_fn = data_fn
        self.n_person = n_person
        self.n_gesture = n_gesture
        self.n_repetions = n_repetions
        self.data_type = data_type
        self.extension = extension
        self.as_gray = as_gray
        self.isTrain = isTrain
        self.transform = transform
        self.dataAug = dataAug
        self.model = model
        self.person_ids = person_ids

        # Create dir dict
        self.data = {}
        sample = 0
        for p in self.person_ids:  # person loop
            for g in range(self.n_gesture):  # gesture loop
                for r in range(self.n_repetions):  # reps loop
                    self.data[sample] = {}
                    self.data[sample]['p'] = p
                    self.data[sample]['g'] = g
                    self.data[sample]['r'] = r
                    self.data[sample]['fn'] = os.path.join(*[self.data_fn,
                                                             "P"+str(p+1),
                                                             "G"+str(g+1),
                                                             self.data_type,
                                                             str(r+1) + self.extension])
                    sample += 1
        # print(self.data)

    def __len__(self):
        return len(self.data)

    def getDataIndex(self, y, g, rep):
        for sample in range(len(self.data)):
            if (y == self.data[sample]['g'] and
                g == self.data[sample]['p'] and
                rep == self.data[sample]['r']):

                return sample

    def __getitem__(self, index):
        # x (image), y, g (person id), rep
        img_fn = self.data[index]['fn']
        # x = Image.open(img_fn)
        x = io.imread(img_fn)
        h, w, _ = x.shape

        y = self.data[index]['g']
        g = self.data[index]['p']
        rep = self.data[index]['r']

        # 1D one hot encoding
        # labels
        y_1D = one_hot_1D(n_classes=self.n_gesture, label=y)
        # person id
        g_1D = one_hot_1D(n_classes=len(self.person_ids),
                          label=self.person_ids.index(g))

        # 2D one hot encoding
        # labels
        y_2D = one_hot_2D(n_classes=self.n_gesture, size=(h, w),
                          label=y)

        # person id
        g_2D = one_hot_2D(n_classes=len(self.person_ids), size=(h, w),
                          label=self.person_ids.index(g))

        # plt.figure()
        # plt.subplot(121)
        # plt.imshow(x)
        # print("1: ", x.shape)
        # transform
        if self.transform:
            self.tranforms = self.tranformations()
            x = self.tranforms(x)
        # print("2: ", x.shape)
        # plt.subplot(122)
        # plt.imshow(inverse_transform(x))
        # plt.show()

        if self.model == 'cnn':
            return x, y, g
        elif self.model == 'vae':
            return x, y  # , n_person-1
        elif self.model == 'cvae':
            return x, y, g, y_1D, g_1D, y_2D, g_2D
        elif self.model == 'twins':
            # for valiadation and test just return x, y, g
            if not self.isTrain:
                return x, y, g

            # INPUTS FOR PRINCIPAL CVAE DECODER
            # there is 50% of changing id
            p = 0.5
            is2change = np.random.choice([0,1], 1, p=[1-p, p])

            g_decoder = copy.deepcopy(g)
            g_decoder_1D = copy.deepcopy(g_1D)
            x_decoder = copy.deepcopy(x)

            if is2change:
                while g_decoder == g:
                    g_decoder = randint(0, len(self.person_ids)-1)
                g_decoder_1D = one_hot_1D(n_classes=len(self.person_ids),
                                          label=g_decoder)

                sample = self.getDataIndex(y, self.person_ids[g_decoder],
                                           randint(0, self.n_repetions-1))

                img_fn = self.data[sample]['fn']
                # x_decoder = Image.open(img_fn)
                x_decoder = io.imread(img_fn)

                # transform
                if self.transform:
                    self.tranforms = self.tranformations()
                    x_decoder = self.tranforms(x_decoder)

            # INPUTS FOR EXTRA CVAE ENCODER
            # generate person id g2 (g2 != g)
            g2 = copy.deepcopy(g)
            g2_2D = copy.deepcopy(g_2D)
            while g2 == g:
                g2 = random.randint(0, len(self.person_ids)-1)

            g2_2D = one_hot_2D(n_classes=len(self.person_ids), size=(h, w),
                               label=g2)

            sample = self.getDataIndex(y, self.person_ids[g2],
                                       randint(0, self.n_repetions-1))

            img_fn = self.data[sample]['fn']
            # x2 = Image.open(img_fn)
            x2 = io.imread(img_fn)

            # transform
            if self.transform:
                self.tranforms = self.tranformations()
                x2 = self.tranforms(x2)

            return(x, y, g, y_1D, g_1D, y_2D, g_2D,
                   x_decoder, g_decoder, g_decoder_1D,
                   x2, g2, g2_2D)

    def tranformations(self):
        # celeba transforms
        if not self.dataAug:
            data_transform = transforms.Compose([Normalize_0_1(),
                                                 Normalize_1_1(),
                                                 numpyToTensor()])
        else:
            # print('INNNN')
            data_transform = transforms.Compose([Normalize_0_1(),
                                                 ImgAugTransform(),
                                                 randomColourTransform(),
                                                 Normalize_1_1(),
                                                 numpyToTensor()])
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
