import torch

def l2normalize(v: torch.Tensor, axis=0, eps=1e-12) -> torch.Tensor:
    return v / (v.norm(dim=axis) + eps)

def get_layer(model: torch.nn.Module, name: str) -> torch.nn.Module:
    layer = model
    for attr in name.split("."):
        layer = getattr(layer, attr)
    return layer


def set_layer(model: torch.nn.Module, name: str, layer: torch.nn.Module) -> None:
    try:
        attrs, name = name.rsplit(".", 1)
        model = get_layer(model, attrs)
    except ValueError as e:
        print(e)
    setattr(model, name, layer)

@torch.no_grad()
def singular_norm(input_tensor: torch.Tensor, power_iterations: int=50) -> torch.Tensor:

    B, S, F = input_tensor.shape

    factory_kwargs = {"dtype": input_tensor.dtype, "device": input_tensor.device}

    sigmas = torch.empty(size=(B, ), **factory_kwargs, requires_grad=False)
    v, u = torch.randn(size=(B, S), **factory_kwargs, 
                       requires_grad=False), torch.randn(size=(B, F), **factory_kwargs, 
                                                         requires_grad=False)
    v, u = l2normalize(v), l2normalize(u)

    for _ in range(power_iterations):
        for b in range(B):
            w = input_tensor[b, :, :]
            v[b, :] = l2normalize(torch.matmul(w, u[b, :]))
            u[b, :] = l2normalize(torch.matmul(w.T, v[b, :]))
    for b in range(B):
        sigmas[b] = torch.matmul(
            torch.matmul(v[b,:], input_tensor[b, :, :]), u[b, :]
        )
        
    return sigmas


def trace_loss(x: torch.Tensor, ) -> torch.Tensor:
    B, S, D = x.size()
    x = x - x.mean(dim=1, keepdim=True)
    frob = x.pow(2).sum(dim=(1,2))
    tl = (torch.bmm(x.permute(0, 2, 1), x) / frob[:, None, None] * D)\
        .diagonal(offset=0, dim1=-1, dim2=-2).add(-1).pow(2).sum(dim=-1)
    return tl.mean(dim=0)