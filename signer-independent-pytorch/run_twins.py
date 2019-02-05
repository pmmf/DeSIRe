import argparse
import os
import sys

sys.path.insert(0, './utils/')
sys.path.insert(0, './losses/')
sys.path.insert(0, './layers/')
sys.path.insert(0, './data/')
sys.path.insert(0, './models/')

import numpy as np
import torch
import copy

from data import CelebA, KinectLeap, split_data, getSplitter
from twins import TWINS
from torchsummary import summary
from torchvision import transforms

import matplotlib.pyplot as plt
plt.switch_backend('agg')

from losses import twins_loss, cross_entropy_loss
from utils import merge_images, inverse_transform, annealing_function, tsne

import pickle

MODEL_LIST = {'twins': TWINS}
DATASETS_LIST = {'celeba': CelebA, 'staticSL': KinectLeap}
BATCH_SIZE = 32
LEARNING_RATE = 1e-04
CNN_WEIGHT = 1
L2_REG = 1e-05
TWINS_REG = 0.01
CVAE_WEIGHT = 0.1
KL_WEIGHT = 8e-02
IM_SIZE = (3, 100, 100)
EPOCHS = 150
ATR_LABEL = 5
ATR_ID = 9 + 10
DATASET_SIZE = {'staticSL': KinectLeap}
SPLITS = 5
MODE = 'groups'


def save_twins_plots(model, data_loader, device, plot_fn):
    with torch.no_grad():  # we do not need gradients
        model.eval()

        (x, y, _, y_1D, _, y_2D, g_2D,
            x_decoder, _, g_decoder_1D,
            x2, _, g2_2D) = list(data_loader)[0]

        # send mini-batch to gpu
        x = x.to(device)
        y = y.to(device)
        # g = g.to(device)
        y_1D = y_1D.to(device)
        # g_1D = g_1D.to(device)
        y_2D = y_2D.to(device)
        g_2D = g_2D.to(device)
        x_decoder = x_decoder.to(device)
        # g_decoder = g_decoder.to(device)
        g_decoder_1D = g_decoder_1D.to(device)
        x2 = x2.to(device)
        # g2 = g2.to(device)
        g2_2D = g2_2D.to(device)

        # encoding decoding
        for pi, p_id in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 13]):
            ggg = torch.zeros(BATCH_SIZE, 11).to(device)
            ggg[:, pi] = 1
            z_mean, z_log_var = model.cvae.encoder(x, y_2D, g_2D)
            x_rec = model.cvae.decoder(z_mean, y_1D, ggg)
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
            plt.savefig(os.path.join(*(plot_fn, 'twins_enc_dec' + str(p_id) + '.png')))  # save the figure to file
            plt.close()

        # reconstructions
        (x_reconst, z_mean, z_log_var,
            z_mean2, z_log_var2,
            h1, y_pred) = model(x, y, y_1D, g_decoder_1D, y_2D, g_2D,
                                x2, g2_2D)

        org_images = merge_images(inverse_transform(x.detach()), (4, 8))
        rec_images = merge_images(inverse_transform(x_reconst.detach()), (4, 8))
        # var.detach().numpy()
        plt.figure()
        plt.subplot(121)
        plt.imshow(org_images)
        plt.axis('off')
        plt.subplot(122)
        plt.imshow(rec_images)
        plt.axis('off')
        plt.show()
        plt.savefig(os.path.join(*(plot_fn, 'twins_rec.png')))  # save the figure to file
        plt.close()

        # new examples
        z = torch.randn(BATCH_SIZE, 128).to(device)
        yyy = torch.zeros(BATCH_SIZE, 10).to(device)
        ggg = torch.zeros(BATCH_SIZE, 11).to(device)
        yyy[:, ATR_LABEL] = 1
        ggg[:, 10] = 1
        new_rec = model.cvae.decoder(z, yyy, ggg)
        new_images = merge_images(inverse_transform(new_rec.detach()), (4, 8))
        plt.figure()
        plt.imshow(new_images)
        plt.axis('off')
        plt.show()
        plt.savefig(os.path.join(*(plot_fn, 'twins_new.png')))  # save the figure to file
        plt.close()


def plot_train_history(train_history, basedir='.'):
    best_val_epoch = np.argmin(train_history['class_loss_val'])
    best_val_acc = train_history['acc_val'][best_val_epoch]
    best_val_loss = train_history['class_loss_val'][best_val_epoch]
    epochs = len(train_history['loss_tr'])
    x = range(epochs)

    # training loss (total loss)
    plt.figure()
    plt.plot(x, train_history['loss_tr'], 'r-')
    plt.xlabel('Epoch')
    plt.ylabel('Total train loss')
    plt.axis([0, epochs, 0, max(train_history['loss_tr'])])
    plt.savefig(os.path.join(basedir, 'train_loss.png'))

    # classification accuracies and losses
    plt.figure()
    plt.subplot(311)
    plt.plot(x, train_history['class_loss_tr'], 'r-')
    plt.plot(x, train_history['class_loss_val'], 'g-')
    plt.plot(best_val_epoch, best_val_loss, 'bx')
    plt.xlabel('Epoch')
    plt.ylabel('Classif. loss')
    plt.legend(['train', 'valid'])
    plt.axis([0, epochs, 0, max(train_history['class_loss_tr'])])
    plt.subplot(312)
    plt.plot(x, train_history['acc_tr'], 'r-')
    plt.plot(x, train_history['acc_val'], 'g-')
    plt.plot(best_val_epoch, best_val_acc, 'bx')
    plt.xlabel('Epoch')
    plt.ylabel('Classif. accuracy')
    plt.legend(['train', 'val'])
    plt.axis([0, epochs, 0, 1])
    plt.subplot(313)
    plt.plot(x, train_history['class_w'], 'b-')
    plt.xlabel('Epoch')
    plt.ylabel('Classif. weight')
    plt.axis([0, epochs, 0, max(train_history['class_w'])])
    plt.savefig(os.path.join(basedir, 'classif_loss.png'))

    # embedding regularization loss
    plt.figure()
    plt.plot(x, train_history['emb_loss'], 'r-')
    plt.xlabel('Epoch')
    plt.ylabel('Embedding loss')
    plt.axis([0, epochs, 0, max(train_history['emb_loss'])])
    plt.savefig(os.path.join(basedir, 'emb_loss.png'))

    # CVAE losses
    plt.figure()
    plt.subplot(411)
    plt.plot(x, train_history['cvae_loss'], 'b-')
    plt.plot(x, train_history['cvae_reconst'], 'r-')
    plt.xlabel('Epoch')
    plt.ylabel('CVAE losses')
    plt.legend(['total', 'reconst', 'kl prior', 'kl_ids'])
    plt.axis([0, epochs, 0, max(train_history['cvae_loss'])])

    plt.subplot(412)
    plt.plot(x, train_history['cvae_kl_ids'], 'g-')
    plt.xlabel('Epoch')
    plt.ylabel('KL ids loss')
    plt.axis([0, epochs, 0, max(train_history['cvae_kl_ids'])])

    plt.subplot(413)
    plt.plot(x, train_history['cvae_kl_prior'], 'g-')
    plt.xlabel('Epoch')
    plt.ylabel('KL prior loss')
    plt.axis([0, epochs, 0, max(train_history['cvae_kl_prior'])])

    plt.subplot(414)
    plt.plot(x, train_history['kl_w'], 'b-')
    plt.xlabel('Epoch')
    plt.ylabel('KL prior weight')
    plt.axis([0, epochs, 0, max(train_history['kl_w'])])
    plt.savefig(os.path.join(basedir, 'cvae_loss.png'))


def fit(model, data, device, output):
    global CNN_W_ANNEAL, KL_W_ANNEAL

    # train and validation loaders
    train_loader, valid_loader = data
    print("Train/Val batches: {}/{}".format(len(train_loader),
                                            len(valid_loader)))

    # Set the optimizer
    optimizer = torch.optim.Adam([{'params': model.cvae.parameters()},
                                  {'params': model.cnn.parameters(),
                                   'weight_decay': L2_REG}],
                                 lr=LEARNING_RATE, weight_decay=0.)

    # Start training
    train_history = {'loss_tr': [], 'cvae_loss': [],
                     'cvae_reconst': [], 'cvae_kl_prior': [],
                     'cvae_kl_ids': [], 'cnn_val': [],
                     'reg_tr': [], 'class_loss_tr': [],
                     'emb_loss': [], 'acc_tr': [],
                     'class_loss_val': [], 'acc_val': [],
                     'class_w': [], 'kl_w': []}

    # Best validation params
    best_val = float('inf')
    best_epoch = 0

    # Start training
    step = 0
    for epoch in range(EPOCHS):
        print('Epoch {}/{}'.format(epoch + 1, EPOCHS))

        # TRAINING
        # set model to train
        model.train()
        for i, (x, y, _, y_1D, _, y_2D, g_2D,
                x_decoder, _, g_decoder_1D,
                x2, _, g2_2D) in enumerate(train_loader):

            # send mini-batch to gpu
            x = x.to(device)
            y = y.to(device)
            # g = g.to(device)
            y_1D = y_1D.to(device)
            # g_1D = g_1D.to(device)
            y_2D = y_2D.to(device)
            g_2D = g_2D.to(device)
            x_decoder = x_decoder.to(device)
            # g_decoder = g_decoder.to(device)
            g_decoder_1D = g_decoder_1D.to(device)
            x2 = x2.to(device)
            # g2 = g2.to(device)
            g2_2D = g2_2D.to(device)

            # print(x.size(), y.size(), y_1D.size(), y_2D.size(), g_2D.size())

            # forward pass
            (x_reconst, z_mean, z_log_var, z,
             z_mean2, z_log_var2,
             h1, y_pred) = model(x, y, y_1D, g_decoder_1D, y_2D, g_2D,
                                 x2, g2_2D)

            # weight annealing
            x0 = 75 * len(train_loader) * BATCH_SIZE
            CNN_W_ANNEAL = annealing_function(init_weight=CNN_WEIGHT,
                                              ftype='exponential',
                                              step=step, k=0.0025,
                                              x0=x0)
            x0 = 30 * len(train_loader) * BATCH_SIZE
            KL_W_ANNEAL = annealing_function(init_weight=KL_WEIGHT,
                                             ftype='exponential',
                                             step=step, k=0.0025,
                                             x0=x0)
            # Compute twins loss
            (loss,
             cvae_loss, cvae_reconst_loss, cvae_kl_prior, cvae_kl_ids,
             class_loss, emb_loss) = twins_loss(x_decoder, y,
                                                x_reconst, z_mean, z_log_var, z,
                                                z_mean2, z_log_var2,
                                                h1, y_pred,
                                                kl_weight=KL_W_ANNEAL,
                                                cvae_weight=CVAE_WEIGHT,
                                                class_weight=CNN_W_ANNEAL,
                                                t_reg=TWINS_REG)

            # Backprop and optimize
            optimizer.zero_grad()  # clear previous gradients
            loss.backward()        # compute new gradients
            optimizer.step()       # optimize the parameters

            # display the mini-batch loss
            print('........{}/{} mini-batch loss: {:.3f} |'
                  .format(i + 1, len(train_loader), loss.item()) +
                  ' cvae loss: {:.3f} |'
                  .format(cvae_loss.item()) +
                #   ' cvae loss: {:.3f}/{:.3f}/{:.3f} |'
                #   .format(cvae_loss[0].item(), cvae_loss[1].item(), cvae_loss[2].item()) +
                  ' emb loss: {:.3f} |'
                  .format(emb_loss.item()) +
                  ' class loss: {:.3f}'
                  .format(class_loss.item()), flush=True, end='\r')

            step += len(x)

        # Evaluation phase
        (loss_tr,
         cvae_loss, cvae_reconst, cvae_kl_prior, cvae_kl_ids,
         class_loss_tr, emb_loss, acc_tr) = train_eval(model,
                                                       train_loader,
                                                       device)
        train_history['loss_tr'].append(loss_tr.item())
        train_history['cvae_loss'].append(cvae_loss.item())
        train_history['cvae_reconst'].append(cvae_reconst.item())
        train_history['cvae_kl_prior'].append(cvae_kl_prior.item())
        train_history['cvae_kl_ids'].append(cvae_kl_ids.item())
        train_history['class_loss_tr'].append(class_loss_tr.item())
        train_history['emb_loss'].append(emb_loss.item())
        train_history['acc_tr'].append(acc_tr)

        train_history['class_w'].append(CNN_W_ANNEAL)
        train_history['kl_w'].append(KL_W_ANNEAL)

        # print('VALIDATION')
        class_loss_val, acc_val = valid_eval(model, valid_loader, device)

        # train_history['loss_val'].append(loss_val)
        # train_history['cvae_val'].append(cvae_val)
        train_history['class_loss_val'].append(class_loss_val.item())
        # train_history['reg_val'].append(reg_val)
        train_history['acc_val'].append(acc_val)

        # save best validation model
        if best_val > class_loss_val:
            torch.save(model.state_dict(), os.path.join(*(output, 'twins.pth')))
            best_val = class_loss_val
            best_epoch = epoch

        # display the training loss
        print()
        print('>> Train loss: {:.5f} |'.format(loss_tr.item()) +
              ' cvae loss: {:.5f} |'.format(cvae_loss.item()) +
              ' class loss: {:.5f} |'.format(class_loss_tr.item()) +
              ' emb loss: {:.5f} |'.format(emb_loss.item()) +
              ' acc: {:.5f}'.format(acc_tr))

        print('>> Valid loss: {:.5f} |'.format(class_loss_val.item()) +
              ' cvae loss: {} |'.format('-') +
              ' class loss: {:.5f} |'.format(class_loss_val.item()) +
              ' reg loss: {} |'.format('-') +
              ' acc: {:.5f}'.format(acc_val))
        print('>> WEIGHTS - KL:{:.5f}/CVAE:{:.5f}/CNN:{:.5f}/TWINS_REG:{:.5f}'.format(KL_W_ANNEAL,
              CVAE_WEIGHT, CNN_W_ANNEAL, TWINS_REG))
        print('>> Best model: {}/{:.5f}'.format(best_epoch+1, best_val))
        print()

    # save train/valid history
    # plot_fn = os.path.join(*(output, 'cnn_history.png'))
    # save_twins_plots(model, train_loader, device, output)

    # make training plots (losses, accuracies, etc)
    plot_train_history(train_history, basedir=output)

    # return best validation model
    model.load_state_dict(torch.load(os.path.join(*(output, 'twins.pth'))))

    return model, train_history, valid_loader


def valid_eval(model, data_loader, device):
    global CNN_W_ANNEAL, KL_W_ANNEAL

    with torch.no_grad():
        # set model to train
        model.eval()
        cnn_eval = 0
        N = 0
        n_correct = 0
        for i, (x, y, g) in enumerate(data_loader):
            # send mini-batch to gpu
            x = x.to(device)
            y = y.to(device)
            # g = g.to(device)

            # forward pass (INFERENCE: JUST ON CLASSIFIER)
            h1, y_pred = model.cnn(x)

            # Compute twins loss
            loss_cnn = cross_entropy_loss(y_pred, y)

            # Acumulate mini-batches losses
            cnn_eval += loss_cnn * x.shape[0]

            # Compute Acc
            N += x.shape[0]
            ypred_ = torch.argmax(y_pred, dim=1)
            n_correct += torch.sum((ypred_ == y).float()).item()

        # Average losses and acc
        cnn_eval = cnn_eval / N  # (i+1)
        acc = n_correct / N

        return cnn_eval, acc


def train_eval(model, data_loader, device):
    global CNN_W_ANNEAL, KL_W_ANNEAL

    with torch.no_grad():
        # set model to train
        model.eval()
        loss_eval = 0
        cvae_eval = 0
        cvae_reconst_eval = 0
        cvae_kl_prior_eval = 0
        cvae_kl_ids_eval = 0
        class_loss_eval = 0
        emb_loss_eval = 0
        N = 0
        n_correct = 0
        for i, (x, y, g, y_1D, _, y_2D, g_2D,
                x_decoder, _, g_decoder_1D,
                x2, _, g2_2D) in enumerate(data_loader):
            # print('>> ', np.unique(g))
            # send mini-batch to gpu
            x = x.to(device)
            y = y.to(device)
            # g = g.to(device)
            y_1D = y_1D.to(device)
            # g_1D = g_1D.to(device)
            y_2D = y_2D.to(device)
            g_2D = g_2D.to(device)
            x_decoder = x_decoder.to(device)
            # g_decoder = g_decoder.to(device)
            g_decoder_1D = g_decoder_1D.to(device)
            x2 = x2.to(device)
            # g2 = g2.to(device)
            g2_2D = g2_2D.to(device)

            # forward pass
            (x_reconst, z_mean, z_log_var, z,
             z_mean2, z_log_var2,
             h1, y_pred) = model(x, y, y_1D, g_decoder_1D, y_2D, g_2D,
                                 x2, g2_2D)

            # Compute twins loss
            (loss,
             cvae_loss, cvae_reconst_loss, cvae_kl_prior, cvae_kl_ids,
             class_loss, emb_loss) = twins_loss(x_decoder, y,
                                                x_reconst, z_mean, z_log_var, z,
                                                z_mean2, z_log_var2,
                                                h1, y_pred,
                                                kl_weight=KL_W_ANNEAL,
                                                cvae_weight=CVAE_WEIGHT,
                                                class_weight=CNN_W_ANNEAL,
                                                t_reg=TWINS_REG)

            # Acumulate mini-batches losses
            loss_eval += loss * x.shape[0]
            cvae_eval += cvae_loss * x.shape[0]
            cvae_reconst_eval += cvae_reconst_loss * x.shape[0]
            cvae_kl_prior_eval += cvae_kl_prior * x.shape[0]
            cvae_kl_ids_eval += cvae_kl_ids * x.shape[0]
            class_loss_eval += class_loss * x.shape[0]
            emb_loss_eval += emb_loss * x.shape[0]

            # Compute Acc
            N += x.shape[0]
            ypred_ = torch.argmax(y_pred, dim=1)
            n_correct += torch.sum((ypred_ == y).float()).item()

        # Average losses and acc
        loss_eval /= N
        cvae_eval /= N
        cvae_reconst_eval /= N
        cvae_kl_prior_eval /= N
        cvae_kl_ids_eval /= N
        class_loss_eval /= N
        emb_loss_eval /= N
        acc = n_correct / N

        return loss_eval, cvae_eval, cvae_reconst_eval, cvae_kl_prior_eval, cvae_kl_ids_eval, class_loss_eval, emb_loss_eval, acc


def main():
    # set random seed
    np.random.seed(42)
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True

    # Parsing arguments
    parser = argparse.ArgumentParser(description='signer-independent project')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--output', default='./output_twins')

    args = parser.parse_args()

    # Make output direcotiry if not exists
    if not os.path.isdir(args.output):
        os.mkdir(args.output)

    # select gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # dataset
    dataset = DATASETS_LIST[args.dataset](model=args.model)
    X_to_split = np.zeros((len(dataset), 1))
    print(len(dataset))

    # get data splitter
    dataSplitter = getSplitter(dataset, n_splits=SPLITS, mode=MODE,
                               test_size=.10)

    results = []
    split = 0

    for split, (tr_indexes, test_indexes) in enumerate(dataSplitter):
        output_fn = os.path.join(args.output, 'split_' + str(split))

        if not os.path.isdir(output_fn):
            os.mkdir(output_fn)

        print('Split', split)

        # split data
        (train_loader,
         valid_loader,
         test_loader) = split_data(dataset,
                                   (tr_indexes, test_indexes),
                                   BATCH_SIZE,
                                   dataAug=True,
                                   mode='groups')

        # Initialize the model
        model = MODEL_LIST[args.model](input_shape=IM_SIZE).to(device)
        print(model)

        # Train or test
        if args.mode == 'train':
            # Fit model
            model, _, valid_loader = fit(model=model,
                                         data=(train_loader, valid_loader),
                                         device=device,
                                         output=output_fn)
        elif args.mode == 'test':
            model.load_state_dict(torch.load(
                                  os.path.join(*(output_fn, 'twins.pth'))))

        # Test results
        test_loss, test_acc = valid_eval(model, test_loader, device)
        print('##!!!! Test loss: {:.5f} |'.format(test_loss.item()) +
              ' Test Acc: {:.5f}'.format(test_acc))

        results.append((test_loss.item(), test_acc))

        # TSNE maps
        # on train
        tsne(model.cvae, train_loader, device,
             plot_fn=os.path.join(*(output_fn, 'tsne_train.png')))
        # on test
        tsne(model.cnn, test_loader, device,
             plot_fn=os.path.join(*(output_fn, 'tsne_test.png')))

    # save results
    print(results)
    # asdas
    res_fn = os.path.join(args.output, 'res.pckl')
    pickle.dump(results, open(res_fn, "wb"))
    results = pickle.load(open(res_fn, "rb"))

    print(results)


if __name__ == '__main__':
    main()
