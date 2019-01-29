import argparse
import os
import sys

import numpy as np
import torch

from data.data import CelebA, KinectLeap, split_data
from models.cvae import CVAE
from torchsummary import summary
from torchvision import transforms

import matplotlib.pyplot as plt
plt.switch_backend('agg')

from losses.losses import vae_loss
from utils.utils import merge_images, inverse_transform

MODEL_LIST = {'cvae': CVAE}
DATASETS_LIST = {'celeba': CelebA, 'staticSL': KinectLeap}
BATCH_SIZE = 32
LEARNING_RATE = 1e-03
KL_WEIGHT = 1e-04  # 1e-03
IM_SIZE = (3, 100, 100)
EPOCHS = 200
ATR_LABEL = 5
ATR_ID = 9 + 10


def samples_from_id(data_loader, person_id=None):
    x_list = []
    y_list = []
    for i, (x, y, g, *tail) in enumerate(data_loader):
        if g == person_id:
            x_list += x
            y_list += y

    return np.array(x_list), np.array(y_list)

def eval_model(model, data_loader, device):
    with torch.no_grad():
        # set model to train
        model.eval()
        loss_eval = 0
        reconst_eval = 0
        kl_eval = 0
        for i, (x, y, g, y_1D, g_1D, y_2D, g_2D) in enumerate(data_loader):
            # send mini-batch to gpu
            x = x.to(device)
            y_1D = y_1D.to(device)
            y_2D = y_2D.to(device)
            g_1D = g_1D.to(device)
            g_2D = g_2D.to(device)

            # forward pass
            r_mean, z_mean, z_log_var = model(x, y_1D, g_1D, y_2D, g_2D)

            # Compute vae loss
            loss, reconst_loss, kl_loss = vae_loss(x, r_mean, z_mean,
                                                   z_log_var,
                                                   kl_weight=KL_WEIGHT)
            loss_eval += loss
            reconst_eval += reconst_loss
            kl_eval += kl_loss

        loss_eval = loss_eval/(i+1)
        reconst_eval = reconst_eval/(i+1)
        kl_eval = kl_eval/(i+1)

        return loss_eval, reconst_eval, kl_eval


def main():
    # set random seed
    np.random.seed(42)

    # Parsing arguments
    parser = argparse.ArgumentParser(description='signer-independent project')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--output', default='/data/pmmf/junk')

    args = parser.parse_args()

    # Make output direcotiry if not exists
    if not os.path.isdir(args.output):
        os.mkdir(args.output)

    # select gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # dataset
    dataset = DATASETS_LIST[args.dataset](model=args.model)
    print("Dataset size: ", len(dataset))
    # for i in range(100, len(dataset)), 10:
    # #     X, y, g, y_, y__, g_, g__ = dataset[i]
    # #     X = inverse_transform(X)
    #     print(y)
    #     input()
    #     print(g)
    #     print(y_)
    #     print(g_)
    #     print(y__)
    #     print(g__)
    #     plt.figure()
    #     plt.imshow(X)
    #     plt.axis('off')
    #     plt.show()
    # y = [dataset[i][1] for i in range(len(dataset))]
    # print(y)

    # Split train and validation
    train_loader, valid_loader = split_data(dataset, BATCH_SIZE,
                                            groups=True)
    print("Train/Val batches: {}/{}".format(len(train_loader),
                                            len(valid_loader)))

    # Fit model
    # Initialize the model
    model = MODEL_LIST[args.model](input_shape=IM_SIZE).to(device)
    print(model)
    # summary(model, [IM_SIZE, (10,), (10, 100, 100)])

    # Set the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Start training
    for epoch in range(EPOCHS):
        print('Epoch {}/{}'.format(epoch + 1, EPOCHS))

        # TRAINING
        # set model to train
        model.train()
        for i, (x, _, _, y_1D, g_1D, y_2D, g_2D) in enumerate(train_loader):
            # send mini-batch to gpu
            x = x.to(device)
            y_1D = y_1D.to(device)
            y_2D = y_2D.to(device)
            g_1D = g_1D.to(device)
            g_2D = g_2D.to(device)

            # forward pass
            r_mean, z_mean, z_log_var = model(x, y_1D, g_1D, y_2D, g_2D)

            # Compute vae loss
            loss, reconst_loss, kl_loss = vae_loss(x, r_mean, z_mean,
                                                   z_log_var,
                                                   kl_weight=KL_WEIGHT)

            # Backprop and optimize
            optimizer.zero_grad()  # clear previous gradients
            loss.backward()        # compute new gradients
            optimizer.step()       # optimize the parameters

            # display the mini-batch loss
            print('........{}/{} mini-batch loss: {:.3f} |'
                  .format(i + 1, len(train_loader), loss.item()) +
                  ' reconst loss: {:.3f} |'
                  .format(reconst_loss.item()) +
                  ' kl loss: {:.3f}'
                  .format(kl_loss.item()), flush=True, end='\r')

        # Validation
        tr_loss, tr_recLoss, tr_klLoss = eval_model(model, train_loader,
                                                    device)
        val_loss, val_recLoss, val_klLoss = eval_model(model, valid_loader,
                                                       device)

        # display the training loss
        print()
        print('>> Train loss: {:.5f} |'.format(tr_loss.item()) +
              ' reconst loss: {:.5f} |'.format(tr_recLoss.item()) +
              ' kl loss: {:.5f}'.format(tr_klLoss.item()))

        print('>> Valid loss: {:.5f} |'.format(val_loss.item()) +
              ' reconst loss: {:.5f} |'.format(val_recLoss.item()) +
              ' kl loss: {:.5f}'.format(val_klLoss.item()))

        print()
        # print()
        # print('saving ...')
        with torch.no_grad():  # we do not need gradients
            model.eval()

            # encoding decoding
            ggg = torch.zeros(BATCH_SIZE, 14).to(device)
            ggg[:, 10] = 1
            x, _, _, y_1D, g_1D, y_2D, g_2D = list(valid_loader)[0]
            x = x.to(device)
            y_1D = y_1D.to(device)
            y_2D = y_2D.to(device)
            g_1D = g_1D.to(device)
            g_2D = g_2D.to(device)

            z_mean, z_log_var = model.encoder(x, y_2D, g_2D)
            x_rec = model.decoder(z_mean, y_1D, ggg)
            org_images = merge_images(inverse_transform(x.detach()), (4, 8))
            rec_images = merge_images(inverse_transform(x_rec.detach()), (4, 8))
            # var.detach().numpy()
            plt.figure()
            plt.subplot(121)
            plt.imshow(org_images)
            plt.axis('off')
            plt.subplot(122)
            plt.imshow(rec_images)
            plt.axis('off')
            plt.show()
            plt.savefig(os.path.join(*(args.output, 'cvae_enc_dec.png')))  # save the figure to file
            plt.close()

            # reconstructions
            r_mean, _, _ = model(x, y_1D, g_1D, y_2D, g_2D)
            org_images = merge_images(inverse_transform(x.detach()), (4, 8))
            rec_images = merge_images(inverse_transform(r_mean.detach()), (4, 8))
            # var.detach().numpy()
            plt.figure()
            plt.subplot(121)
            plt.imshow(org_images)
            plt.axis('off')
            plt.subplot(122)
            plt.imshow(rec_images)
            plt.axis('off')
            plt.show()
            plt.savefig(os.path.join(*(args.output, 'cvae_rec.png')))  # save the figure to file
            plt.close()

            # new examples
            z = torch.randn(BATCH_SIZE, 128).to(device)
            yyy = torch.zeros(BATCH_SIZE, 10).to(device)
            ggg = torch.zeros(BATCH_SIZE, 14).to(device)
            yyy[:, ATR_LABEL] = 1
            ggg[:, 10] = 1
            new_rec = model.decoder(z, yyy, ggg)
            new_images = merge_images(inverse_transform(new_rec.detach()), (4, 8))
            plt.figure()
            plt.imshow(new_images)
            plt.axis('off')
            plt.show()
            plt.savefig(os.path.join(*(args.output, 'cvae_new.png')))  # save the figure to file
            plt.close()


if __name__ == '__main__':
    main()
