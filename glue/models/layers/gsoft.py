import torch
import torch.nn as nn
import math
import torch.nn.functional as F

from models.layers.gs_orthogonal import GSOrthogonal


class GSOFTLayer(nn.Module):
    def __init__(
            self,
            pre_layer: nn.Module,
            in_features: int,
            out_features: int,
            nblocks: int,
            orthogonal: bool = True,
            method: str = 'cayley',
            block_size = None,
            scale: bool = True
            ):

        super().__init__()

        self.pre_layer = pre_layer
        self.in_features = in_features
        self.out_features = out_features
        self.nblocks = nblocks
        self.scale = scale

        self.gs_ort = GSOrthogonal(in_features, nblocks, orthogonal, method, block_size)
        
        if self.scale:
            self.gsoft_s = nn.Parameter(torch.ones(out_features))
        

    def forward(self, x: torch.Tensor):
        
        x = self.gs_ort(x)
        x = F.linear(x, self.pre_layer.weight)
        
        if self.scale:
            x = self.gsoft_s * x
        
        if self.pre_layer.bias is not None:
            x = x + self.pre_layer.bias
        
        return x