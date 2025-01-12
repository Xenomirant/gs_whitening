# code modified from https://github.com/singlasahil14/SOC
import logging
import os
import time
import hydra
import wandb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lip_convnets import LipConvNet
from utils import *


logger = logging.getLogger(__name__)


def init_model(args):
    if args.dataset == 'cifar10':
        num_classes = 10    
    elif args.dataset == 'cifar100':
        num_classes = 100
    
    if isinstance(args.groups, str):
        groups = tuple(map(int, args.groups.split()))
    else:
        groups = args.groups

    model = LipConvNet(args.conv_layer, args.activation, init_channels=args.init_channels, 
                       block_size=args.block_size, num_classes=num_classes, 
                       groups=groups, paired=args.paired)
    return model


@hydra.main(config_path="conf", config_name="config_robust", version_base=None)
def main(args):
    args.out_dir += '_' + str(args.dataset) 
    args.out_dir += '_' + str(args.block_size) 
    args.out_dir += '_' + str(args.conv_layer)
    args.out_dir += '_' + str(args.init_channels)
    args.out_dir += '_' + str(args.activation)
    args.out_dir += '_' + str(args.groups)
    
    
    os.makedirs(args.out_dir, exist_ok=True)

    logfile = os.path.join(args.out_dir, 'output.log')
    if os.path.exists(logfile):
        os.remove(logfile)

    logging.basicConfig(
        format='%(message)s',
        level=logging.INFO,
        filename=os.path.join(args.out_dir, 'output.log'))
    logger.info(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    train_loader, test_loader = get_loaders(
        args.data_dir, args.batch_size, args.dataset
    )

    std = cifar10_std

    torch.backends.cudnn.benchmark = True
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
        notes=args.notes,
        tags=[args.dataset, f"lipconvnet-{args.block_size*5}"],
        name=f"{args.conv_layer}, groups={groups}",
        config={
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "model_name": args.model_name,
            "dataset": args.dataset,
            "seed": args.seed,
            "conv_layer": args.conv_layer,
            "min_lr": args.lr_min,
            "max_lr": args.lr_max,
            "weight_decay": args.weight_decay,
            "momentum": args.momentum,
            "activation": args.activation,
            "number of parameters with grad": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "number of all parameters": sum(p.numel() for p in model.parameters()),
            "paired": bool(args.paired),
            "groups": groups
        }
    )

    opt = torch.optim.SGD(
        model.parameters(),
        weight_decay=args.weight_decay,
        lr=args.lr_max,
        momentum=args.momentum
    )

    criterion = nn.CrossEntropyLoss()

    lr_steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=[lr_steps // 2, (3 * lr_steps) // 4], gamma=0.1
    )

    best_model_path = os.path.join(args.out_dir, 'best.pth')
    last_model_path = os.path.join(args.out_dir, 'last.pth')
    last_opt_path = os.path.join(args.out_dir, 'last_opt.pth')
    
    # Training
    std = torch.tensor(std).cuda()
    L = 1/torch.max(std)
    prev_robust_acc = 0.
    start_train_time = time.time()
    logger.info('Epoch \t Seconds \t LR \t Train Loss \t Train Acc \t Test Loss \t ' + 
                'Test Acc \t Test Robust \t Test Cert')
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

            opt.zero_grad()
            ce_loss.backward()
            opt.step()
            curr_correct = (output.max(1)[1] == y)

            train_loss += ce_loss.item() * y.size(0)
            train_acc += curr_correct.sum().item()
            train_n += y.size(0)
            scheduler.step()
            
        # Check current test accuracy of model
        test_loss, test_acc, mean_cert, robust_acc = evaluate_certificates(test_loader, model, L)

        # robust_acc = test_robust_acc_list[0]
        if (robust_acc >= prev_robust_acc):
            torch.save(model.state_dict(), best_model_path)
            prev_robust_acc = robust_acc
            best_epoch = epoch
        
        epoch_time = time.time()
        lr = scheduler.get_last_lr()[0]
        logger.info(
            '%d \t %.1f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f',
            epoch,
            epoch_time - start_epoch_time,
            lr,
            train_loss / train_n,
            train_acc / train_n, 
            test_loss,
            test_acc,
            robust_acc,
            mean_cert
        )

        wandb.log({
            "train_acc": train_acc / train_n,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "epoch_time": epoch_time - start_epoch_time,
            "test_robust_acc": robust_acc,
            "test_cert": mean_cert
        })
        
        torch.save(model.state_dict(), last_model_path)
        
        trainer_state_dict = { 'epoch': epoch, 'optimizer_state_dict': opt.state_dict()}
        torch.save(trainer_state_dict, last_opt_path)
        
    train_time = time.time()

    logger.info('Total train time: %.4f minutes', (train_time - start_train_time)/60)
    
    
    # Evaluation at best model (early stopping)
    model_test = init_model(args).cuda()
    model_test.load_state_dict(torch.load(best_model_path))
    model_test.float()
    model_test.eval()
        
    start_test_time = time.time()
    test_loss, test_acc, mean_cert, robust_acc = evaluate_certificates(test_loader, model_test, L)
    total_time = time.time() - start_test_time
    
    logger.info("Best Epoch \t Test Loss \t Test Acc \t Robust Acc \t  Mean" "Cert \t Test Time")
    logger.info('%d \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f', best_epoch, test_loss, test_acc, robust_acc, mean_cert, total_time)

    # Evaluation at last model
    model_test.load_state_dict(torch.load(last_model_path))
    model_test.float()
    model_test.eval()

    start_test_time = time.time()
    test_loss, test_acc, mean_cert, robust_acc = evaluate_certificates(test_loader, model_test, L)
    total_time = time.time() - start_test_time
    
    logger.info("Last Epoch \t Test Loss \t Test Acc \t Robust Acc \t  Mean" "Cert \t Test Time")
    logger.info('%d \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f', epoch, test_loss, test_acc, robust_acc, mean_cert, total_time)
    wandb.finish()

if __name__ == "__main__":
    main()