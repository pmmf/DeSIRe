import torch
from torch.nn import functional as F

cross_entropy_loss = F.cross_entropy


def triplet_loss(anchor, positive, negative, ALPHA=0.01):
    pos_dist = torch.mean((anchor - positive)**2, 1)
    neg_dist = torch.mean((anchor - negative)**2, 1)

    basic_loss = (pos_dist - neg_dist) + ALPHA
    loss = torch.mean(torch.max(basic_loss, 0.0), 0)
    return loss


def twins_reg(z_mean, z_mean2, h1):
    sim_loss_cvae = torch.mean((z_mean - z_mean2)**2)  # TODO: KL divergence here
    sim_loss_cnn = torch.mean((z_mean.detach() - h1)**2)
    return sim_loss_cvae + sim_loss_cnn


def twins_loss(x, y,
               x_reconst, z_mean, z_log_var,
               z_mean2,
               h1, y_pred,
               kl_weight=1e-03,
               cvae_reg=1e-03,
               cnn_reg=1,
               t_reg=1e-03):

    # loss of main cvae
    (cvae_loss,
     cvae_reconst_loss,
     cvae_kl_loss) = vae_loss(x, x_reconst, z_mean, z_log_var,
                              kl_weight=kl_weight)

    # cvae_loss_tuple = (cvae_loss, cvae_reconst_loss, cvae_kl_loss)

    # cnn loss
    loss_cnn = cross_entropy_loss(y_pred, y)

    # twins reg
    reg_loss = twins_reg(z_mean, z_mean2, h1)

    # total loss
    loss = cvae_reg*cvae_loss + cnn_reg*loss_cnn + t_reg*reg_loss

    return loss, cvae_loss, loss_cnn, reg_loss


def vae_loss(x, x_reconstructed, z_mean, z_log_var, kl_weight=1e-03):
    # Reconstruction loss
    reconst_loss = torch.mean((x_reconstructed - x)**2)

    # KL loss
    kl_loss = torch.mean(- 0.5 * torch.sum(1 + z_log_var - z_mean**2 -
                         torch.exp(z_log_var), dim=1), dim=0)

    # VAE loss
    loss = reconst_loss + kl_weight * kl_loss

    return loss, reconst_loss, kl_loss


if __name__ == '__main__':
    pass
