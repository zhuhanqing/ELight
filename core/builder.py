'''
Author: Hanqing Zhu(hqzhu@utexas.edu)
Date: 1969-12-31 18:00:00
LastEditTime: 2022-04-09 00:37:07
LastEditors: Hanqing Zhu(hqzhu@utexas.edu)
Description: Build functions
FilePath: /projects/ELight/core/builder.py
'''
import os
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from pyutils.config import configs
from pyutils.datasets import get_dataset
from pyutils.typing import DataLoader, Optimizer, Scheduler
from torch.types import Device

from core.models import *
from pyutils.optimizer.radam import RAdam

__all__ = [
    "make_dataloader",
    "make_model",
    "make_weight_optimizer",
    "make_arch_optimizer",
    "make_optimizer",
    "make_scheduler",
    "make_criterion",
]


def make_dataloader(name: str = None) -> Tuple[DataLoader, DataLoader]:
    name = (name or configs.dataset.name).lower()
    train_dataset, test_dataset = get_dataset(
        name,
        configs.dataset.img_height,
        configs.dataset.img_width,
        dataset_dir=configs.dataset.root,
        transform=configs.dataset.transform,
    )
    validation_dataset = None

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=configs.run.batch_size,
        shuffle=int(configs.dataset.shuffle),
        pin_memory=True,
        num_workers=configs.dataset.num_workers,
    )

    validation_loader = (
        torch.utils.data.DataLoader(
            dataset=validation_dataset,
            batch_size=configs.run.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=configs.dataset.num_workers,
        )
        if validation_dataset is not None
        else None
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=configs.run.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=configs.dataset.num_workers,
    )

    return train_loader, validation_loader, test_loader


def make_model(device: Device, random_state: int = None, model_cfg = None) -> nn.Module:
    model_name = model_cfg.name
    if "vgg" in model_name.lower():
        model = PCMVGG(
            model_name,
            ### unique parameters
            n_class=configs.dataset.num_classes,
            in_channel=configs.dataset.in_channels,
            block_size=model_cfg.block_size,
            in_bit=configs.quantization.input_bit,
            w_bit=configs.quantization.weight_bit,
            mode=model_cfg.mode,
            ### quantization
            input_quant_method=configs.quantization.input_quant_method,
            weight_quant_method=configs.quantization.weight_quant_method,
            p=configs.quantization.quant_noise,
            ### loss_flag
            loss_fn=configs.criterion.diff_loss_fn,
            ### PCM-based PTC
            assign=configs.PTC.assign,
            hasZero=configs.PTC.hasZero,
            pcm_l=configs.PTC.loss_factor,
            device=device
        ).to(device)
        model.reset_parameters()
    else:
        model = None
        raise NotImplementedError(f"Not supported model name: {model_name}")

    return model


def make_optimizer(params, name: str = None, configs=None) -> Optimizer:
    if name == "sgd":
        optimizer = torch.optim.SGD(
            params,
            lr=configs.lr,
            momentum=configs.momentum,
            weight_decay=configs.weight_decay,
            nesterov=True,
        )
    elif name == "adam":
        optimizer = torch.optim.Adam(
            params,
            lr=configs.lr,
            weight_decay=configs.weight_decay,
            betas=getattr(configs, "betas", (0.9, 0.999)),
        )
    elif name == "radam":
        optimizer = RAdam(params, lr=configs.lr, weight_decay=configs.weight_decay
        )
    else:
        raise NotImplementedError(name)

    return optimizer


def make_scheduler(optimizer: Optimizer, name: str = None) -> Scheduler:
    name = (name or configs.scheduler.name).lower()
    if name == "constant":
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1)
    elif name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(configs.run.n_epochs), eta_min=float(configs.scheduler.lr_min)
        )
    elif name == "exp":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=configs.scheduler.lr_gamma)
    elif name == "step":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[82, 122], gamma=0.1)
    else:
        raise NotImplementedError(name)

    return scheduler


def make_criterion(name: str = None) -> nn.Module:
    name = (name or configs.criterion.name).lower()
    if name == "nll":
        criterion = nn.NLLLoss()
    elif name == "mse":
        criterion = nn.MSELoss()
    elif name == "mae":
        criterion = nn.L1Loss()
    elif name == "ce":
        criterion = nn.CrossEntropyLoss()
    else:
        raise NotImplementedError(name)
    return criterion