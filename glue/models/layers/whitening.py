import abc
import torch
import torch.nn as nn
import einops
from torch import Tensor
from models.utils import singular_norm
from typing import Any, Optional


class Whitening2d(nn.Module):
    def __init__(self, 
        num_features,
        iterations=4, 
        use_running_stats_train=True,
        use_batch_whitening=False,
        use_only_running_stats_eval=False,
        track_running_stats: bool = True,
        momentum: Optional[float] = 0.1,
        affine: bool = True,
        device=None,
        dtype=None,
                ):
        super(Whitening2d, self).__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.num_features=num_features
        self.iterations=iterations
        self.use_batch_whitening=use_batch_whitening
        self.use_running_stats_train=use_running_stats_train
        self.use_only_running_stats_eval=use_only_running_stats_eval
        self.track_running_stats=track_running_stats
        self.momentum=momentum
        self.affine=affine

        if self.affine:
            self.weight=torch.nn.Parameter(torch.empty(num_features, **factory_kwargs))
            self.bias=torch.nn.Parameter(torch.empty(num_features, **factory_kwargs))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        
        if self.track_running_stats:
            self.register_buffer(
                "running_mean", torch.zeros(num_features, **factory_kwargs)
            )
            self.register_buffer(
                "running_covariance", torch.eye(num_features, **factory_kwargs)
            )
            self.register_buffer(
                "running_whitening", torch.eye(num_features, **factory_kwargs)
            )
            self.running_mean: Optional[Tensor]
            self.running_covariance: Optional[Tensor]
            self.running_whitening: Optional[Tensor]
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_covariance", None)
            self.register_buffer("running_whitening", None)
        
        self.register_buffer("attention_mask", None)
        self.reset_parameters()

        
    def reset_running_stats(self) -> None:
        if self.track_running_stats:
            self.running_mean.zero_()  
            self.running_covariance.fill_(1)
            self.running_whitening.fill_(1)

    def reset_parameters(self) -> None:
        self.reset_running_stats()
        if self.affine:
            torch.nn.init.ones_(self.weight)
            torch.nn.init.zeros_(self.bias)

    def update_running_statistic(self, running_statistic, value):
        cur = getattr(self, running_statistic,)
        setattr(self, running_statistic, 
                (1-self.momentum)*cur + self.momentum*value
                )

    def forward_train(self, x, attention_mask):
        
        batch_size, w_dim = x.size(0), x.size(-1)
        
        m_r = x.mean(1, keepdim=True)
        if self.use_running_stats_train:
            m = (1-self.momentum)*self.running_mean + self.momentum*m_r
        else:
            m = m_r
        
        xn = x - m

        eye, sigma_r = self.calc_eye_sigma(xn, w_dim=w_dim, batch_size=batch_size)
        if self.use_running_stats_train:
            sigma = (1-self.momentum)*self.running_covariance[None, :, :] + self.momentum*sigma_r
        else:
            sigma = sigma_r

        wh_matrix = self.whiten_matrix(sigma=sigma, eye=eye)

        if self.track_running_stats:
            self.update_running_statistic("running_mean", m_r.mean(dim=0))
            self.update_running_statistic("running_covariance", sigma_r.mean(dim=0))
            self.update_running_statistic("running_whitening", wh_matrix.mean(dim=0))

        decorrelated = torch.bmm(xn, wh_matrix)
        return decorrelated
    
    @torch.no_grad
    def forward_test(self, x, attention_mask):

        batch_size, w_dim = x.size(0), x.size(-1)

        if self.use_only_running_stats_eval:
            xn = x - self.running_mean
            decorrelated = torch.bmm(xn, 
                                     einops.repeat(self.running_whitening, "feats1 feats2 -> batch feats1 feats2", batch=batch_size), 
                                     )
            return decorrelated
        
        m = x.mean(1, keepdim=True)
        m = (1-self.momentum)*self.running_mean + self.momentum*m

        xn = x - m

        eye, sigma = self.calc_eye_sigma(xn, w_dim=w_dim, batch_size=batch_size)
        sigma = (1-self.momentum)*self.running_covariance[None, :, :] + self.momentum*sigma

        wh_matrix = self.whiten_matrix(sigma=sigma, eye=eye)
        decorrelated = torch.bmm(xn, wh_matrix)
        return decorrelated

    def forward(self, x, **kwargs):
        
        attention_mask = getattr(self, "attention_mask")
        print(attention_mask)

        if self.training:
            x = self.forward_train(x=x)
            x = self.weight*x + self.bias
            return x
        x = self.forward_test(x=x)
        x = self.weight*x + self.bias
        return x

    @abc.abstractmethod
    def whiten_matrix(self, sigma, eye):
        pass

    def calc_eye_sigma(self, xn, w_dim, batch_size):
        eye = einops.repeat(torch.eye(w_dim).type(xn.type()), 
                "feats1 feats2 -> batch feats1 feats2", batch=batch_size).to(xn.device)
        if self.use_batch_whitening:
            batch_cov = einops.rearrange(xn, "batch sequence feats -> (batch sequence) feats")
            sigma = einops.einsum(batch_cov, batch_cov, 
                                  "batch_seq feats1, batch_seq feats2 -> feats1 feats2") / (batch_cov.shape[0] - 1)
            sigma = einops.repeat(sigma, "feats1 feats2 -> batch feats1 feats2", batch=batch_size)
        else:
            sigma = einops.einsum(xn, xn, 
                                  "batch seq feats1, batch seq feats2 -> batch feats1 feats2") / (xn.shape[1] - 1)
        return eye, sigma

    def extra_repr(self):
        return (
            "{num_features}, iterations={iterations}, momentum={momentum}, affine={affine}, "
            "track_running_stats={track_running_stats}, use_batch_whitening={use_batch_whitening}, "
            "use_running_stats_train={use_running_stats_train}, use_only_running_stats_eval={use_only_running_stats_eval}".format(**self.__dict__)
        )

class Whitening2dIterNorm(Whitening2d):

    def whiten_matrix(self, sigma, eye):
        trace = sigma.diagonal(offset=0, dim1=-2, dim2=-1).sum(-1)
        trace = trace.reshape(sigma.size(0), 1, 1)
        sigma_norm = sigma * trace.reciprocal()

        projection = eye
        for _ in range(self.iterations):
            projection = torch.baddbmm(projection, torch.matrix_power(projection, 3), sigma_norm, beta=1.5, alpha=-0.5)
        wm = projection.mul_(trace.reciprocal().sqrt())
        return wm
    