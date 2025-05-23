import torch

def l2normalize(v, axis=0, eps=1e-12):
    return v / (v.norm(dim=axis) + eps)

def get_layer(model, name):
    layer = model
    for attr in name.split("."):
        layer = getattr(layer, attr)
    return layer


def set_layer(model, name, layer):
    try:
        attrs, name = name.rsplit(".", 1)
        model = get_layer(model, attrs)
    except ValueError as e:
        print(e)
    setattr(model, name, layer)

@torch.no_grad()
def singular_norm(input_tensor: torch.Tensor, power_iterations=20) -> torch.Tensor:

    B, S, F = input_tensor.shape

    factory_kwargs = {"dtype": input_tensor.dtype, "device": input_tensor.device}

    sigmas = torch.empty(size=(B, ), **factory_kwargs)
    v, u = torch.randn(size=(B, S), **factory_kwargs), torch.randn(size=(B, F), **factory_kwargs)
    v, u = l2normalize(v), l2normalize(u)

    for _ in range(power_iterations):
        for b in range(B):
            w = input_tensor[b, :, :]
            v[b] = l2normalize(torch.matmul(w, u[b]))
            u[b] = l2normalize(torch.matmul(w.T, v[b]))
    for b in range(B):
        sigmas[b] = torch.matmul(
            torch.matmul(v[b], input_tensor[b, :, :]), u[b]
        )
        
    return sigmas