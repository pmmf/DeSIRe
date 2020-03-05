import torch

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
    D = mu0.shape[1]

    dkl = .5*(torch.sum(torch.exp(logsigma0 - logsigma1)
                        + torch.exp(-logsigma1)*(mu1 - mu0)**2
                        + logsigma1 - logsigma0, dim=1) - D)

    return dkl


# do some tests to verify correctness
M = 1024
D = 128

mu0 = torch.randn(M, D)
logsigma0 = torch.randn(M, D)
mu1 = torch.randn(M, D)
logsigma1 = torch.randn(M, D)

dkl = kl_div(mu0, logsigma0, mu1, logsigma1)
if torch.any(dkl < 0):
    print('ERROR: Negative KL divergence!\n', dkl)
else:
    print('OK: Positive KL divergence')

dkl = kl_div(mu0, logsigma0, mu0, logsigma0)
if torch.any(dkl != 0):
    print('ERROR: Equal distributions yield non-zero KL divergence!\n', dkl)
else:
    print('OK: Equal distributions yield zero KL divergence')
