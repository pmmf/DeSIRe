import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset
import h5py
import numpy as np
from glob import glob
import os
from skimage import io
from PIL import Image
import matplotlib.pyplot as plt
from skimage import io
from torchsummary import summary

def merge_images(images, size):
    # merge all output images(of sample size:8*8 output images of size 64*64) into one big image
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images): # idx=0,1,2,...,63
        i = idx % size[1] # column number
        j = idx // size[1] # row number
        img[j*h:j*h+h, i*w:i*w+w, :] = image
    return img

def inverse_transform(x):
    x = x.to('cpu').numpy()
    
    x = x.transpose((0, 2, 3, 1))
    x = (x+1.)/2.
    return x       
    

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        sample = sample.transpose((2, 0, 1))
        sample = torch.from_numpy(sample).float()/127.5 - 1.
        return sample
        
class CelebA(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.data_fn = '/data/DB/celebA/img_align_celeba_crop/'
        self.imgs_list = sorted(os.listdir(self.data_fn))

    def __len__(self):
        return len(self.imgs_list)

    def __getitem__(self, index):
        img = io.imread(os.path.join(*(self.data_fn, self.imgs_list[index])))

        if self.transform is not None:
            img = self.transform(img)
            
        return img

class KinectLeap(Dataset):
    def __init__(self, 
                 data_fn='/data/DB/kinect_leap_dataset_signer_independent/',
                 n_person=14,
                 n_gesture=10,
                 n_repetions=10,
                 extension='_rgb.png',
                 data_type='RGB_CROPS_RZE_DISTTR2', 
                 cmap=-1,
                 validation=None, 
                 transform=None):

        self.data_fn = data_fn
        self.n_person = n_person
        self.n_gesture = n_gesture
        self.n_repetions = n_repetions
        self.data_type = data_type
        self.extension = extension
        self.cmap = cmap
        self.validation = validation
        self.transform = transform
        
        self.imgs_list = sorted(os.listdir(self.data_fn))

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

        # read image
        img_fn = os.path.join(*[self.data_fn, 
                                "P" + str(n_person), 
                                "G" + str(n_gesture), 
                                self.data_type, 
                                str(n_rep) + self.extension])
        print(index, img_fn)
        img = io.imread(img_fn)

        # transform
        if self.transform is not None:
            img = self.transform(img)
            
        return img, n_gesture-1, n_person-1



class VAE(nn.Module):
    def __init__(self, 
                 input_shape=(3, 64, 64),
                 base_filters=64, 
                 z_dim=128,
                 kernel_size=5,
                 learning_rate=1e-03,
                 batch_size=64,
                 KLD_weight=5e-04):
        
        super(VAE, self).__init__()

        self.input_shape = input_shape
        self.base_filters = base_filters
        self.z_dim = z_dim
        self.kernel_size = kernel_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.KLD_weight = KLD_weight

        # encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=self.kernel_size, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2))

        # mean and var
        self.fc_mean = nn.Linear(4*4*512, self.z_dim)
        self.fc_var = nn.Linear(4*4*512, self.z_dim)
        self.bn_mean = nn.BatchNorm1d(self.z_dim)
        self.bn_var = nn.BatchNorm1d(self.z_dim)
          
        # decoder
        self.fc_d1 = nn.Linear(self.z_dim, 8*8*256)
        self.bn_d1 = nn.BatchNorm2d(256*8*8)
        self.lr_d1 = nn.LeakyReLU(negative_slope=0.2)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=self.kernel_size, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.ConvTranspose2d(128, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.2),
            nn.ConvTranspose2d(32, 3, kernel_size=5, stride=1, padding=2, output_padding=0),
            # nn.BatchNorm2d(512), # NO BNORM ON LAST LAYER
            nn.Tanh())

            
        
    def encode(self, x):
        enc = self.encoder(x)
        enc = enc.reshape(enc.size(0), -1)
        z_mean = self.bn_mean(self.fc_mean(enc))
        z_log = F.softplus(self.bn_var(self.fc_var(enc))) + 1e-06

        return z_mean, z_log
    
    def reparameterize(self, z_mean, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        return z_mean + eps * std

    def decode(self, z):
        h = self.fc_d1(z)
        h = h.view(-1, 256, 8, 8)
        h = self.lr_d1(self.bn_d1(h))
        h = self.decoder(h)
        return h
    
    def forward(self, x):
        # encode
        z_mean, log_var = self.encode(x)

        # reparameterization trick
        z = self.reparameterize(z_mean, log_var)

        # decode
        x_reconst = self.decode(z)
        return x_reconst, z_mean, log_var

# def plain_vae_loss(inputs, r_mean, z_mean, z_log_var_sq):
#     loss_reconstruction = torch.mean(torch.square(r_mean - inputs), axis=-1)
#     loss_KL = torch.mean(- 0.5 * K.sum(1 + z_log_var_sq - K.square(z_mean) - K.exp(z_log_var_sq), axis=1), axis=0)
#     return loss_reconstruction + (self.KLD_weight * loss_KL)


if __name__ == '__main__':

    dataset = KinectLeap()

    print(0, dataset.index2dataset(0))
    print(10, dataset.index2dataset(10))
    print(100, dataset.index2dataset(100))
    print(101, dataset.index2dataset(101))
    print(751, dataset.index2dataset(751))
    print(1000, dataset.index2dataset(1000))
    print(1001, dataset.index2dataset(1001))
    print(1399, dataset.index2dataset(1399))



    print(len(dataset))
    print(dataset[0])
    for i in range(len(dataset)):
        X, y, group = dataset[i]
        print(y, group)
        plt.figure()
        plt.imshow(X)
        plt.axis('off')
        plt.show()
        # print(tsfrm(celeba_data[i]).shape)


    adasd
    BATCH_SIZE = 64
    num_epochs = 30
    KLD_weight = 5e-04

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    ## DATA
    tsfrm = ToTensor()
    celeba_data = CelebA(tsfrm)

    # creating data indices for training and validation splits
    dataset_size = len(celeba_data)  # number of samples in training + validation sets
    indices = list(range(dataset_size))
    split = int(np.floor(0.2 * dataset_size))  # no. samples in valid set
    np.random.seed(42)
    np.random.shuffle(indices)
    train_indices, valid_indices = indices[split:], indices[:split]

    print(len(train_indices), len(valid_indices))

    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
    valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_indices)

    train_loader = torch.utils.data.DataLoader(celeba_data, batch_size=BATCH_SIZE,
                                            shuffle=False, num_workers=4,
                                            sampler=train_sampler)

    valid_loader = torch.utils.data.DataLoader(celeba_data, batch_size=BATCH_SIZE,
                                            shuffle=False, num_workers=4,
                                            sampler=valid_sampler)




    ## MODEL
    model = VAE().to(device)
    print(model)
    summary(model, (3, 64, 64))
    
    model = VAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-03)

    # Start training
    for epoch in range(num_epochs):
        model.train()
        for i, x in enumerate(train_loader):
            # Forward pass
            x = x.to(device)

            # print("IN: ", x.shape)
            r_mean, z_mean, z_log_var = model(x)
            
            # Compute reconstruction loss and kl divergence
            reconst_loss = torch.mean((r_mean - x)**2)
            loss_KL = torch.mean(- 0.5 * torch.sum(1 + z_log_var - z_mean**2 - torch.exp(z_log_var), dim=1), dim=0)

            
            loss = reconst_loss + (KLD_weight * loss_KL)
            # print(loss)
            
            # Backprop and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i+1) % 50 == 0:
                # print(loss, reconst_loss, loss_KL)
                print ("Epoch[{}/{}], Step [{}/{}], VAE Loss: {:.4f}, Reconst Loss: {:.4f}, KL Div: {:.4f}" 
                    .format(epoch+1, num_epochs, i+1, len(train_loader), loss, reconst_loss, loss_KL))

            if (i+1) % 500 == 0:
                org_images = merge_images(inverse_transform(x), (8, 8))
                rec_images = merge_images(inverse_transform(r_mean.detach()), (8, 8))
                # var.detach().numpy()
                plt.figure()
                plt.subplot(121)
                plt.imshow(org_images)
                plt.axis('off')
                plt.subplot(122)
                plt.imshow(rec_images)
                plt.axis('off')
                plt.show()

                z = torch.randn(BATCH_SIZE, 128).to(device)
                new_rec = model.decode(z)
                new_images = merge_images(inverse_transform(new_rec.detach()), (8, 8))
                plt.figure()
                plt.imshow(new_images)
                plt.axis('off')
                plt.show()


                
            
        model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
        with torch.no_grad():
            loss = 0
            reconst_loss = 0
            loss_KL = 0
            for i, x in enumerate(valid_loader):
                # Forward pass
                x = x.to(device)

                # print("IN: ", x.shape)
                r_mean, z_mean, z_log_var = model(x)
                
                # Compute reconstruction loss and kl divergence
                reconst_loss = torch.mean((r_mean - x)**2)
                loss_KL = torch.mean(- 0.5 * torch.sum(1 + z_log_var - z_mean**2 - torch.exp(z_log_var), dim=1), dim=0)

                loss += reconst_loss + (KLD_weight * loss_KL)
                loss_KL += loss_KL
                reconst_loss += reconst_loss
                # print(i, loss)

            
            print ("[VALID], VAE Loss: {:.4f}, Reconst Loss: {:.4f}, KL Div: {:.4f}".format(loss/(i+1), reconst_loss/(i+1), loss_KL/(i+1)))

            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    ada
    tsfrm = ToTensor()
    celeba_data = CelebA()
    print(len(celeba_data))
    print(celeba_data[0])
    for i in range(len(celeba_data)):
        plt.figure()
        plt.imshow(celeba_data[i])
        plt.axis('off')
        plt.show()
        print(tsfrm(celeba_data[i]).shape)

    pass
