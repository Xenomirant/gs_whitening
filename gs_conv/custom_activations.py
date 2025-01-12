import torch
import torch.nn as nn

import numpy as np


class MinMax(nn.Module):
    def __init__(self):
        super(MinMax, self).__init__()

    def forward(self, z, axis=1):
        a, b = z.split(z.shape[axis] // 2, axis)
        c, d = torch.min(a, b), torch.max(a, b)
        return torch.cat([c, d], dim=axis)

class MinMaxPermuted(nn.Module):
    def __init__(self):
        super(MinMaxPermuted, self).__init__()
    
    def forward(self, z):
        a, b, = z[:, ::2], z[:, 1::2]
        c, d = torch.min(a, b), torch.max(a, b)
        permutation = torch.zeros(z.shape[1], dtype=torch.int, device=z.device)
        permutation[::2] = torch.arange(z.shape[1] // 2, dtype=torch.int)
        permutation[1::2] = torch.arange(z.shape[1] // 2, dtype=torch.int) + z.shape[1]// 2
        return torch.cat([c, d], dim=1)[:, permutation]