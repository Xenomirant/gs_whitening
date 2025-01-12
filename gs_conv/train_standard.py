# code modified from https://github.com/singlasahil14/SOC
import logging
import os
import time
import wandb
import numpy as np
import torch
import torch.nn as nn
import hydra
from preactresnet import *
from utils import (
    get_loaders,
    evaluate_standard,
)

logger = logging.getLogger(__name__)


def init_model(args):
    if args.dataset == "cifar10":
        num_classes = 10
    elif args.dataset == "cifar100":
        num_classes = 100

    model_func = resnet_mapping[args.model_name]
    if isinstance(args.groups, str):
        groups = tuple(map(int, args.groups.split()))
    else:
        groups = args.groups

    model = model_func(
        conv_name=args.conv_layer,
        activation_name=args.activation,
        num_classes=num_classes,
        groups=groups
    )
    return model


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(args):
    args.out_dir += "_" + str(args.dataset)
    args.out_dir += "_" + str(args.model_name)
    args.out_dir += "_" + str(args.conv_layer)
    args.out_dir += "_" + str(args.activation)
    args.out_dir += "_" + str(args.groups)
    args.out_dir += "_" + str(args.weight_decay)

    os.makedirs(args.out_dir, exist_ok=True)

    logfile = os.path.join(args.out_dir, "output.log")
    if os.path.exists(logfile):
        os.remove(logfile)

    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO,
        filename=os.path.join(args.out_dir, "output.log"),
    )
    logger.info(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    train_loader, test_loader = get_loaders(
        args.data_dir, args.batch_size, args.dataset
    )
    model = init_model(args).cuda()
    model.train()
    if isinstance(args.groups, str):
        groups = tuple(map(int, args.groups.split()))
    else:
        groups = args.groups
    
    wandb.login(key=args.wandb_key, relogin=True)
    wandb.init(
        entity="",
        project="",
        name=f"{args.model_name}, {args.conv_layer}, {args.dataset}, groups={groups}, wd={args.weight_decay}",
        config={
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr_scheduler": args.lr_scheduler,
            "model_name": args.model_name,
            "dataset": args.dataset,
            "seed": args.seed,
            "activation": args.activation,
            "conv_layer": args.conv_layer,
            "min_lr": args.lr_min,
            "max_lr": args.lr_max,
            "weight_decay": args.weight_decay,
            "momentum": args.momentum,
            "number of parameters with grad": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "number of all parameters": sum(p.numel() for p in model.parameters()),
            "groups": groups
        }
    )

    artifact = wandb.Artifact(
        name="run_config",
        type="config",
    )
    artifact.add_file(f"{hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}/.hydra/config.yaml")
    wandb.log_artifact(
        artifact
    )

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

    opt = torch.optim.SGD(
        [
            {"params": activation_params, "weight_decay": 0.0},
            {
                "params": (conv_params + other_params),
                "weight_decay": args.weight_decay,
            },
        ],
        lr=args.lr_max,
        momentum=args.momentum,
    )

    criterion = nn.CrossEntropyLoss()

    lr_steps = args.epochs * len(train_loader)
    if args.lr_scheduler == "cyclic":
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            opt,
            base_lr=args.lr_min,
            max_lr=args.lr_max,
            step_size_up=lr_steps / 2,
            step_size_down=lr_steps / 2,
        )
    elif args.lr_scheduler == "multistep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            opt, milestones=[lr_steps // 4, (3 * lr_steps) // 4], gamma=0.1
        )

    best_model_path = os.path.join(args.out_dir, "best.pth")
    last_model_path = os.path.join(args.out_dir, "last.pth")
    last_opt_path = os.path.join(args.out_dir, "last_opt.pth")

    prev_test_acc = 0.0
    start_train_time = time.time()
    logger.info(
        "Epoch \t Seconds \t LR \t Train Loss \t Train Acc \t Test Loss \t Test Acc"
    )
    for epoch in range(args.epochs):
        model.train()
        start_epoch_time = time.time()
        train_loss = 0
        train_acc = 0
        train_n = 0
        for i, (X, y) in enumerate(train_loader):
            X, y = X.cuda(), y.cuda()

            output = model(X)
            ce_loss = criterion(output, y)
            wandb.log({
                "train_loss": ce_loss.item(),
                "lr": scheduler.get_last_lr()[0]
            })
            opt.zero_grad(set_to_none=True)
            ce_loss.backward()
            opt.step()

            train_loss += ce_loss.item() * y.size(0)
            train_acc += (output.max(1)[1] == y).sum().item()
            train_n += y.size(0)
            scheduler.step()

        epoch_time = time.time()

        # Check current test accuracy of model
        test_loss, test_acc = evaluate_standard(test_loader, model)
        if test_acc > prev_test_acc:
            torch.save(model.state_dict(), best_model_path)
            prev_test_acc = test_acc
            best_epoch = epoch

        lr = scheduler.get_last_lr()[0]
        logger.info(
            "%d \t %.1f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f",
            epoch,
            epoch_time - start_epoch_time,
            lr,
            train_loss / train_n,
            train_acc / train_n,
            test_loss,
            test_acc,
        )
        wandb.log({
            "train_acc": train_acc / train_n,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "epoch_time": epoch_time - start_epoch_time
        })
        torch.save(model.state_dict(), last_model_path)

        trainer_state_dict = {"epoch": epoch, "optimizer_state_dict": opt.state_dict()}
        torch.save(trainer_state_dict, last_opt_path)

    train_time = time.time()

    logger.info("Total train time: %.4f minutes", (train_time - start_train_time) / 60)

    # Evaluation at early stopping
    model_test = init_model(args).cuda()
    model_test.load_state_dict(torch.load(best_model_path))
    model_test.float()
    model_test.eval()

    start_test_time = time.time()
    test_loss, test_acc = evaluate_standard(test_loader, model_test)
    test_time = time.time()
    logger.info("Best Epoch \t Test Loss \t Test Acc \t Test Time")
    logger.info(
        "%d \t %.4f \t %.4f \t %.4f",
        best_epoch,
        test_loss,
        test_acc,
        (test_time - start_test_time) / 60,
    )

    # Evaluation at last model
    model_test.load_state_dict(torch.load(last_model_path))
    model_test.float()
    model_test.eval()

    start_test_time = time.time()
    test_loss, test_acc = evaluate_standard(test_loader, model_test)
    test_time = time.time()
    logger.info("Last Epoch \t Test Loss \t Test Acc \t Test Time")
    logger.info(
        "%d \t %.4f \t %.4f \t %.4f",
        epoch,
        test_loss,
        test_acc,
        (test_time - start_test_time) / 60,
    )
    wandb.finish()


if __name__ == "__main__":
    main()
