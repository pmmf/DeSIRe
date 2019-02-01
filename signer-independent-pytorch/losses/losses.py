import torch
from torch.nn import functional as F

cross_entropy_loss = F.cross_entropy


def kl_div(mu0, logsigma0, mu1, logsigma1):
    """Computes KL(N0 || N1), i.e. the KL divergence between two normal
    distributions N0 and N1.
    Inputs:
    mu0 - the mean of N0 (tensor with shape (M, D))
    logsigma0 - the diagonal log-covariance of N0 (tensor with shape (M, D))
    mu1 - the mean of N1 (tensor with shape (M, D))
    logsigma1 - the diagonal log-covariance of N1 (tensor with shape (M, D))
    Outputs:
    dkl - the KL divergence (tensor with shape (M,)"""

    # D = mu0.shape[1]

    dkl = .5*(torch.mean(torch.exp(logsigma0 - logsigma1)
                         + torch.exp(-logsigma1)*(mu1 - mu0)**2
                         + logsigma1 - logsigma0, dim=1))

    return dkl


def triplet_loss(anchor, positive, negative, ALPHA=1.0):
    pos_dist = torch.mean((anchor - positive)**2, 1)
    neg_dist = torch.mean((anchor - negative)**2, 1)

    basic_loss = (pos_dist - neg_dist) + ALPHA
    loss = torch.mean(torch.max(basic_loss, 0.0), 0)
    return loss


def twins_loss(x, y,
               x_reconst, z_mean, z_log_var, z,
               z_mean2, z_log_var2,
               h1, y_pred,
               kl_weight=1e-03,
               cvae_weight=1e-03,
               class_weight=1,
               t_reg=1e-03):

    # CVAE loss
    (cvae_loss,
     cvae_reconst_loss,
     cvae_kl_prior) = vae_loss(x, x_reconst, z_mean, z_log_var,
                               kl_weight=kl_weight)

    cvae_kl_ids = torch.mean(kl_div(z_mean, z_log_var, z_mean2, z_log_var2),
                             dim=0)

    cvae_loss += kl_weight*cvae_kl_ids

    # classification loss
    class_loss = cross_entropy_loss(y_pred, y)

    # classification embeddings loss
    class_emb_loss = torch.mean((z.detach() - h1)**2)

    # total loss
    loss = cvae_weight*cvae_loss + class_weight*class_loss + t_reg*class_emb_loss

    return loss, cvae_loss, cvae_reconst_loss, cvae_kl_prior, cvae_kl_ids, class_loss, class_emb_loss


def vae_loss(x, x_reconstructed, z_mean, z_log_var, kl_weight=1e-03):
    # Reconstruction loss
    reconst_loss = torch.mean((x_reconstructed - x)**2)

    # KL loss
    kl_loss = -.5*torch.mean(torch.mean(1 + z_log_var - z_mean**2 -
                                        torch.exp(z_log_var), dim=1), dim=0)

    # VAE loss
    loss = reconst_loss + kl_weight * kl_loss

    return loss, reconst_loss, kl_loss


if __name__ == '__main__':
    pass
