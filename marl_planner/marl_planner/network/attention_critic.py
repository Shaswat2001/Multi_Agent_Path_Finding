import torch
from torch import nn 

class AttentionCritic(nn.Module):

    def __init__(self, args):
        super(AttentionCritic,self).__init__()

        self.args = args
