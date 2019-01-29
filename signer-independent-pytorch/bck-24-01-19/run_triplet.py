import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt


import sys
sys.path.insert(0, './utils/')
sys.path.insert(0, './losses/')
sys.path.insert(0, './layers/')
sys.path.insert(0, './data/')
sys.path.insert(0, './models/')

from data import CelebA, KinectLeap, split_data, getSplitter
from vae import VAE
from cnn import CNN
from utils import merge_images, inverse_transform

from torchsummary import summary
from torchvision import transforms
import pickle

MODEL_LIST = {'vae': VAE, 'cnn': CNN}
DATASETS_LIST = {'celeba': CelebA, 'staticSL': KinectLeap}
BATCH_SIZE = 32
LEARNING_RATE = 1e-04
KL_WEIGHT = 8e-04  # 1e-03
IM_SIZE = (3, 100, 100)
EPOCHS = 75
REG = 1e-04
SPLITS = 5
MODE = 'groups'
TRIPLET_WEIGHT = 0.1

from torch.nn import functional as F
loss_fn = F.cross_entropy
triplet_loss = torch.nn.TripletMarginLoss(margin=1.0, p=2)


def triplet_selection(anchor, model, data_loader, device):
    with torch.no_grad():
        model.eval()

        # batch data
        anchor_x, anchor_y, anchor_g, anchor_rep = anchor
        anchor_x = anchor_x.to(device)
        anchor_y = anchor_y.to(device)
        anchor_g = anchor_g.to(device)
        anchor_rep = anchor_rep.to(device)
        # print("anchor_x: ", anchor_x.size())

        # Get embeddings (forward pass)
        emb_anchor_x = model.feat_extractor(anchor_x).reshape(anchor_x.size(0),
                                                              -1)
        # print("emb_anchor_x: ", emb_anchor_x.size())

        # Create mask of valid triplets
        anchor_y_g_rep = torch.cat((anchor_y.unsqueeze(1),
                                    anchor_g.unsqueeze(1),
                                    anchor_rep.unsqueeze(1)), 1).unsqueeze(1)
        # print("anchor_y_g_rep: ", anchor_y_g_rep.size())
        # print(anchor_y_g_rep)

        mask = anchor_y_g_rep.unsqueeze(1) == anchor_y_g_rep
        A, B, C = mask[:, :, :, 0], mask[:, :, :, 1], mask[:, :, :, 2]
        pos_neg = np.logical_and(np.logical_or(np.logical_not(B),
                                               np.logical_not(C)),
                                 A)[:, :, 0].numpy().astype(bool)

        # print(mask)
        # print("mask: ",  mask.shape)
        # print(pos_neg)
        # print("pos_neg: ",  pos_neg.shape)

        # Compute embeddings l2 distance
        dist_matrix = torch.mean(((emb_anchor_x.unsqueeze(1) -
                                   emb_anchor_x)**2).float(), 2)
        # print("dist_matrix: ", dist_matrix.size())
        argsort_matrix = np.argsort(dist_matrix.to('cpu').numpy(), 1)
        # print("argsort_matrix: ", argsort_matrix[0][pos_neg[0]])

        # Select positive and negative pairs
        positives = np.array([argsort_matrix[idx][i[argsort_matrix[idx]]][-1:]
                              if len(argsort_matrix[idx][i[argsort_matrix[idx]]][-1:]) != 0 else
                              [idx]
                              for idx, i in enumerate(pos_neg)])
        np.fill_diagonal(pos_neg, True)
        negatives = np.array([argsort_matrix[idx][~i[argsort_matrix[idx]]][:1]
                              for idx, i in enumerate(pos_neg)])

        # Select positive and negative pairs
        # positives = np.array([np.argwhere(i == True)[0]
        #                       for idx, i in enumerate(pos_neg)])
        # np.fill_diagonal(pos_neg, True)
        # negatives = np.array([np.argwhere(~i == True)[0]
        #                       for idx, i in enumerate(pos_neg)])

        # positives = positives[:, np.newaxis]
        # print('POSITIVES')
        # print(positives)
        # print(negatives)
        # print(positives.shape)

        # print(anchor_x.size(), positives.shape, negatives.shape)
        x_p = anchor_x[positives, :, :, :]
        x_n = anchor_x[negatives, :, :, :]

        x_p = x_p[:, 0]
        x_n = x_n[:, 0]

        # for iii in range(64):
        #     plt.figure()
        #     plt.subplot(131)
        #     plt.imshow(inverse_transform((anchor_x[iii])))
        #     plt.axis('off')
        #     plt.title('Anchor')
        #     plt.subplot(132)
        #     plt.imshow(inverse_transform((x_p[iii])))
        #     plt.axis('off')
        #     plt.title('Positive')
        #     plt.subplot(133)
        #     plt.imshow(inverse_transform((x_n[iii])))
        #     plt.axis('off')
        #     plt.title('Negative')
        #     plt.show()

        return anchor_x, x_p, x_n


def fit(model, data, device, output):
    # train and validation loaders
    train_loader, valid_loader = data
    print("Train/Val batches: {}/{}".format(len(train_loader),
                                            len(valid_loader)))

    # Set the optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=LEARNING_RATE,
                                 weight_decay=REG)

    # Start training
    train_history = {'train_loss': [], 'train_acc': [],
                     'val_loss': [], 'val_acc': []}

    # Best validation params
    best_val = float('inf')
    best_epoch = 0

    for epoch in range(EPOCHS):
        print('Epoch {}/{}'.format(epoch + 1, EPOCHS))

        # TRAINING
        # set model to train
        model.train()
        for i, (x, y, g, rep) in enumerate(train_loader):

            _, x_pos, x_neg = triplet_selection((x, y, g, rep),
                                                model,
                                                train_loader,
                                                device)

            # send mini-batch to gpu
            x = x.to(device)
            y = y.to(device)
            x_pos = x_pos.to(device)
            x_neg = x_neg.to(device)

            # for iii in range(64):
            #     plt.figure()
            #     plt.subplot(111)
            #     plt.imshow(inverse_transform((x[iii])))
            #     plt.axis('off')
            #     plt.show()
            #     break

            # forward pass
            h1, y_pred = model(x)
            h1_pos, y_pred_pos = model(x_pos)
            h1_neg, y_pred_neg = model(x_neg)

            # Compute vae loss
            t_loss = triplet_loss(h1, h1_pos, h1_neg)
            c_loss = loss_fn(y_pred, y)
            # loss = loss_fn(y_pred, y)
            loss = t_loss*TRIPLET_WEIGHT + c_loss

            # Backprop and optimize
            optimizer.zero_grad()  # clear previous gradients
            loss.backward()        # compute new gradients
            optimizer.step()       # optimize the parameters

            # display the mini-batch loss
            print('........{}/{} mini-batch loss: {:.3f}'
                  .format(i + 1, len(train_loader), loss.item()),
                  flush=True, end='\r')

        # Validation
        tr_loss, tr_acc = eval_model(model, train_loader, device)
        train_history['train_loss'].append(tr_loss)
        train_history['train_acc'].append(tr_acc)

        val_loss, val_acc = eval_model(model, valid_loader, device, debug=True)
        train_history['val_loss'].append(val_loss)
        train_history['val_acc'].append(val_acc)

        # save best validation model
        if best_val > val_loss:
            torch.save(model.state_dict(), os.path.join(*(output, 'cnn.pth')))
            best_val = val_loss
            best_epoch = epoch

        # display the training loss
        print()
        print('>> Train loss: {:.5f} |'.format(tr_loss.item()) +
              ' Train Acc: {:.5f}'.format(tr_acc))

        print('>> Valid loss: {:.5f} |'.format(val_loss.item()) +
              ' Valid Acc: {:.5f}'.format(val_acc))
        print('>> Best model: {}/{:.5f}'.format(best_epoch+1, best_val))
        print()

    # save train/valid history
    plot_fn = os.path.join(*(output, 'cnn_history.png'))
    plot_train_history(train_history, plot_fn=plot_fn)

    # return best validation model
    model.load_state_dict(torch.load(os.path.join(*(output, 'cnn.pth'))))

    return model, train_history, valid_loader


def plot_train_history(train_history, plot_fn=None):
    import matplotlib.pyplot as plt
    plt.switch_backend('agg')

    best_val_epoch = np.argmin(train_history['val_loss'])
    best_val_acc = train_history['val_acc'][best_val_epoch]
    best_val_loss = train_history['val_loss'][best_val_epoch]
    plt.figure(figsize=(7, 5))
    epochs = len(train_history['train_loss'])
    x = range(epochs)
    plt.subplot(211)
    plt.plot(x, train_history['train_loss'], 'r-')
    plt.plot(x, train_history['val_loss'], 'g-')
    plt.plot(best_val_epoch, best_val_loss, 'bx')
    plt.xlabel('Epoch')
    plt.ylabel('Train/Val loss')
    plt.legend(['train_loss', 'val_loss'])
    plt.axis([0, epochs, 0, max(train_history['train_loss'])])
    plt.subplot(212)
    plt.plot(x, train_history['train_acc'], 'r-')
    plt.plot(x, train_history['val_acc'], 'g-')
    plt.plot(best_val_epoch, best_val_acc, 'bx')
    plt.xlabel('Epoch')
    plt.ylabel('Train/Val acc')
    plt.legend(['train_acc', 'val_acc'])
    plt.axis([0, epochs, 0, 1])
    if plot_fn:
        plt.show()
        plt.savefig(plot_fn)
        plt.close()
    else:
        plt.show()


def eval_model(model, data_loader, device, debug=False):
    with torch.no_grad():
        # set model to train
        model.eval()
        loss_eval = 0
        N = 0
        n_correct = 0
        for i, (x, y, g, _) in enumerate(data_loader):
            # send mini-batch to gpu
            x = x.to(device)
            y = y.to(device)

            # if debug:
            #     for iii in range(64):
            #         plt.figure()
            #         plt.subplot(111)
            #         plt.imshow(inverse_transform((x[iii])))
            #         plt.axis('off')
            #         plt.show()
            #         break

            # forward pass
            _, y_pred = model(x)

            # Compute cnn loss
            loss = loss_fn(y_pred, y)
            loss_eval += loss * x.shape[0]

            # Compute Acc
            N += x.shape[0]
            ypred_ = torch.argmax(y_pred, dim=1)
            n_correct += torch.sum(1.*(ypred_ == y)).item()

        loss_eval = loss_eval / N
        acc = n_correct / N
        return loss_eval, acc


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
    parser.add_argument('--output', default='/data/pmmf/signer_independent/triplet_01_FC3')

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
    print(len(dataset[0]))

    # get data splitter
    dataSplitter = getSplitter(dataset, n_splits=SPLITS, mode=MODE,
                               test_size=.10)

    results = []
    split = 0

    for split, (tr_indexes, test_indexes) in enumerate(dataSplitter):
        output_fn = os.path.join(args.output, 'split_' + str(split))

        if not os.path.isdir(output_fn):
            os.mkdir(output_fn)

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
        if args.mode == 'Train':
            # Fit model
            model, _, valid_loader = fit(model=model,
                                         data=(train_loader, valid_loader),
                                         device=device,
                                         output=output_fn)
        elif args.mode == 'Test':
            model.load_state_dict(torch.load(
                                  os.path.join(*(output_fn, 'cnn.pth'))))

        test_loss, test_acc = eval_model(model, test_loader, device)
        print('##!!!! Test loss: {:.5f} |'.format(test_loss.item()) +
              ' Test Acc: {:.5f}'.format(test_acc))

        results.append((test_loss.item(), test_acc))

    # save results
    print(results)
    # asdas
    res_fn = os.path.join(args.output, 'res.pckl')
    pickle.dump(results, open(res_fn, "wb"))
    results = pickle.load(open(res_fn, "rb"))

    print(results)

if __name__ == '__main__':
    main()

    # per-layer lr
    # optim.SGD([
    #             {'params': model.base.parameters()},
    #             {'params': model.classifier.parameters(), 'lr': 1e-3}
    #         ], lr=1e-2, momentum=0.9)

    # def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=7):
    #     """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    #     lr = init_lr * (0.1**(epoch // lr_decay_epoch))

    #     if epoch % lr_decay_epoch == 0:
    #         print('LR is set to {}'.format(lr))

        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = lr

    #     return optimizer




