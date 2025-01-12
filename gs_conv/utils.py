# code modified from https://github.com/singlasahil14/SOC
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import math
import numpy as np
from skew_ortho_conv import SOC, GS_SOC, PermutedSOC, GS_SOC_Accelerated, LPRSOC
from custom_activations import MinMax, MinMaxPermuted

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2507, 0.2507, 0.2507)

mu = torch.tensor(cifar10_mean).view(3, 1, 1).cuda()
std = torch.tensor(cifar10_std).view(3, 1, 1).cuda()


def get_loaders(dir_, batch_size, dataset_name="cifar10", normalize=True):
    if dataset_name == "cifar10":
        dataset_func = datasets.CIFAR10
    elif dataset_name == "cifar100":
        dataset_func = datasets.CIFAR100

    if normalize:
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(cifar10_mean, cifar10_std),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(cifar10_mean, cifar10_std),
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

    num_workers = 4
    train_dataset = dataset_func(
        dir_, train=True, transform=train_transform, download=True
    )
    test_dataset = dataset_func(
        dir_, train=False, transform=test_transform, download=True
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=2,
    )
    return train_loader, test_loader


def evaluate_standard(test_loader, model):
    test_loss = 0
    test_acc = 0
    n = 0
    model.eval()
    with torch.no_grad():
        for i, (X, y) in enumerate(test_loader):
            X, y = X.cuda(), y.cuda()
            output = model(X)
            loss = F.cross_entropy(output, y)
            test_loss += loss.item() * y.size(0)
            test_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
    return test_loss / n, test_acc / n


def evaluate_certificates(test_loader, model, L, epsilon=36.0):
    losses_list = []
    certificates_list = []
    correct_list = []
    model.eval()
    with torch.no_grad():
        for i, (X, y) in enumerate(test_loader):
            X, y = X.cuda(), y.cuda()
            output = model(X)
            loss = F.cross_entropy(output, y, reduction="none")
            losses_list.append(loss)

            output_max, output_amax = torch.max(output, dim=1)

            onehot = torch.zeros_like(output).cuda()
            onehot[torch.arange(output.shape[0]), output_amax] = 1.0

            output_trunc = output - onehot * 1e6

            output_nextmax = torch.max(output_trunc, dim=1)[0]
            output_diff = output_max - output_nextmax

            certificates = output_diff / (math.sqrt(2) * L)
            correct = output_amax == y

            certificates_list.append(certificates)
            correct_list.append(correct)

        losses_array = torch.cat(losses_list, dim=0).cpu().numpy()
        certificates_array = torch.cat(certificates_list, dim=0).cpu().numpy()
        correct_array = torch.cat(correct_list, dim=0).cpu().numpy()

    mean_loss = np.mean(losses_array)
    mean_acc = np.mean(correct_array)

    mean_certificates = (certificates_array * correct_array).sum() / correct_array.sum()

    robust_correct_array = (certificates_array > (epsilon / 255.0)) & correct_array
    robust_correct = robust_correct_array.sum() / robust_correct_array.shape[0]
    return mean_loss, mean_acc, mean_certificates, robust_correct


conv_mapping = {
    "standard": nn.Conv2d,
    "soc": SOC,
    "gs_soc": GS_SOC,
    "gs_soc_accelerated": GS_SOC_Accelerated,
    "permuted_soc": PermutedSOC,
    "lpr_soc": LPRSOC
}


activation_dict = {
    "relu": F.relu,
    "swish": F.silu,
    "sigmoid": F.sigmoid,
    "tanh": F.tanh,
    "softplus": F.softplus,
    "minmax": MinMax(),
    "minmax_permuted": MinMaxPermuted()
}


def activation_mapping(activation_name):
    activation_func = activation_dict[activation_name]
    return activation_func


def parameter_lists(model):
    conv_params = []
    activation_params = []
    other_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "activation" in name:
                activation_params.append(param)
            elif "conv" in name:
                conv_params.append(param)
            else:
                other_params.append(param)
    return conv_params, activation_params, other_params
