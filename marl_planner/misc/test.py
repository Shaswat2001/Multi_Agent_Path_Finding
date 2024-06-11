import numpy as np
import torch
a = torch.zeros((60,3))
b = torch.ones((60,3))
l = [a,b]
print(torch.hstack(l))