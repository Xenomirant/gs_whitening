import abc
import torch
import torch.nn as nn

class Whitening2d(nn.Module):
    def __init__(self, axis=1, iterations=4):
        super(Whitening2d, self).__init__()
        self.axis = axis
        self.iterations = iterations
        # self.correlation_matrix = 

    def forward(self, x):
        assert self.axis in (0,1), "axis must be in (batch,sequence) !"
        
        w_dim = x.size(-1)
        m = x.mean(1 if self.axis == 1 else 0, keepdim=True)
        # m = m.view(1, -1) if self.axis == 1 else m.view(-1, 1)
        xn = x - m 

        eye = torch.eye(w_dim).type(xn.type())[None, :, :].repeat(x.shape[0], 1, 1)

        sigma = torch.bmm(xn.permute(0, 2, 1), xn) / (xn.shape[self.axis] - 1)

        matrix = self.whiten_matrix(sigma, eye)  
        decorrelated = torch.bmm(xn, matrix)

        return decorrelated

    @abc.abstractmethod
    def whiten_matrix(self, sigma, eye):
        pass

    def extra_repr(self):
        return "axis={}, iterations={}".format(
            self.axis, self.iterations
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
    