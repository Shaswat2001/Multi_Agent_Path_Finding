from collections import namedtuple
import numpy as np
import torch
from torch import nn

EPS = 1e-8

PolicyOps = namedtuple('PolicyOps', 'raw_mean mean log_std pi log_prob_pi')

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

class Squash(nn.Module):

    def __init__(self,in_min, in_max, out_min, out_max):
        super(Squash,self).__init__()
        self.input_min = in_min
        self.input_max = in_max
        self.input_scale = in_max - in_min
        self.output_min = out_min
        self.output_max = out_max
        self.output_scale = out_max - out_min
    
    def forward(self, x, **kwargs):
        return (x - self.input_min) / self.input_scale * self.output_scale + self.output_min
    
class DiagonalGaussianSample(nn.Module):

    def __init__(self):
        super(DiagonalGaussianSample,self).__init__()
    
    def forward(self,mean,log_std):
        meanOne = torch.zeros_like(mean)
        stdOne = torch.ones_like(mean)
        eps = torch.normal(meanOne,stdOne)
        std = torch.exp(log_std)
        sample = mean + std * eps
        return sample
    
class TanhDiagonalGaussianLogProb(nn.Module):

    def __init__(self):
        super(TanhDiagonalGaussianLogProb,self).__init__()
    
    def forward(self,gaussian_samples, tanh_gaussian_samples, mean, log_std):

        std = torch.exp(log_std)
        log_probs_each_dim = -0.5 * np.log(2 * np.pi) - log_std - (gaussian_samples - mean) ** 2 / (2 * std ** 2 + EPS)
        log_prob = torch.sum(log_probs_each_dim, dim=0, keepdims=True)
        tanh_gaussian_samples = clip_but_pass_gradient(tanh_gaussian_samples, low=-1, high=1)
        correction = torch.sum(torch.log(1 - tanh_gaussian_samples ** 2 + EPS), dim=0, keepdims=True)
        log_prob -= correction
        return log_prob
    
def clip_but_pass_gradient(x, low=-1., high=1.):
    # From Spinning Up implementation
    clip_up = x > high
    clip_up = clip_up.type(torch.FloatTensor)
    clip_low = x <low
    clip_low = clip_low.type(torch.FloatTensor)
    val = (high - x) * clip_up + (low - x) * clip_low
    return x + val.detach()