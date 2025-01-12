import math
import numpy as np

import torch
from torch.nn import functional as F
import torch.nn as nn

from models.layers.blockdiag_butterfly_multiply import BlockdiagButterflyMultiply


class GSOrthogonal(nn.Module):
    def __init__(self, n: int, nblocks: int, orthogonal=True, method="cayley", block_size=None):

        if block_size is not None:
            assert n % block_size == 0
            nblocks = n // block_size

        assert n % nblocks == 0

        super().__init__()

        self.gsoft_R = nn.Parameter(torch.empty(nblocks, n // nblocks, n // nblocks))
        self.gsoft_L = nn.Parameter(torch.empty(nblocks, n // nblocks, n // nblocks))

        self.orthogonal = orthogonal
        self.n = n
        self.nblocks = nblocks
        self.block_size = n // nblocks
        self.method = method

        self.blockdiag_butterfly_multiply = BlockdiagButterflyMultiply.apply

        self.reset_parameters()

    def reset_parameters(self):
        # initialize whole layer as identity matrix

        if self.orthogonal:
            torch.nn.init.zeros_(self.gsoft_L)
            torch.nn.init.zeros_(self.gsoft_R)

        else:
            block_size = self.n // self.nblocks
            self.gsoft_L.data = torch.eye(block_size).unsqueeze(0).expand(self.nblocks, block_size, block_size)
            self.gsoft_R.data = torch.eye(block_size).unsqueeze(0).expand(self.nblocks, block_size, block_size)
    
    def exp_full(self, data):
        skew = 0.5 * (data - data.transpose(1, 2))
        return torch.matrix_exp(skew)

    def cayley_batch(self, data):
        b, r, c = data.shape
        # Ensure the input matrix is skew-symmetric
        skew = 0.5 * (data - data.transpose(1, 2))
        I = torch.eye(r, device=data.device).unsqueeze(0).expand(b, r, c)

        # Perform the Cayley parametrization
        Q = torch.linalg.solve(I - skew, I + skew, left=False)
        return Q
    
    def forward(self, x):

        if self.orthogonal:
            if self.method == "cayley":
                L = self.cayley_batch(self.gsoft_L)
                R = self.cayley_batch(self.gsoft_R)
            elif self.method == "exp":
                L = self.exp_full(self.gsoft_L)
                R = self.exp_full(self.gsoft_R)
            else:
                raise NotImplementedError("Method is not supported. Use 'cayley' or 'exp'.")
        else:
            L = self.gsoft_L
            R = self.gsoft_R

        return self.blockdiag_butterfly_multiply(x, R, L)
