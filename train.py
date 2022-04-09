'''
Author: Hanqing Zhu(hqzhu@utexas.edu)
Date: 2022-04-07 10:35:16
LastEditTime: 2022-04-09 00:41:32
LastEditors: Hanqing Zhu(hqzhu@utexas.edu)
Description: Train the aging-aware model with write-aware training
FilePath: /projects/ELight/train.py
'''
import argparse
import os
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pyutils.config import configs
from pyutils.general import logger as lg
from pyutils.torch_train import (
    BestKModelSaver,
    count_parameters,
    get_learning_rate,
    load_model,
    save_model,
    set_torch_deterministic,
)
from pyutils.typing import Criterion, DataLoader, Optimizer, Scheduler
from core import builder

def train(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: Optimizer,
    scheduler: Scheduler,
    epoch: int,
    criterion: Criterion,
    device: torch.device,
    teacher: nn.Module = None,
    soft_criterion: Criterion = None,
) -> None:
    model.train()
    step = epoch * len(train_loader)
    correct = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        optimizer.zero_grad()
        output = model(data)

        def _get_loss(output, target):
            if teacher:
                with torch.no_grad():
                    teacher_score = teacher(data).detach()
                loss = soft_criterion(output, teacher_score, target)
            else:
                loss = criterion(output, target)
            return loss

        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum()

        loss = _get_loss(output, target)
        class_loss = loss
        
        if configs.criterion.diff_loss_weight > 0:
            if (configs.criterion.diff_loss_fn == 'l1'):
                if (configs.criterion.block_match_method == 'meanG'):
                    diff_loss = configs.criterion.diff_loss_weight * model.get_difference_loss_global_L1(True)
                if (configs.criterion.block_match_method == 'meanR'):
                    diff_loss = configs.criterion.diff_loss_weight * model.get_difference_loss_row_L1(True)
                if (configs.criterion.block_match_method == 'neighbor'):
                    raise NotImplementedError
                    diff_loss = configs.criterion.diff_loss_weight * model.get_difference_loss_row_L1_nei(True)
            elif (configs.criterion.diff_loss_fn == 'l2'):
                if (configs.criterion.block_match_method == 'meanG'):
                    diff_loss = configs.criterion.diff_loss_weight * model.get_difference_loss_global_L2(True)
                if (configs.criterion.block_match_method == 'meanR'):
                    diff_loss = configs.criterion.diff_loss_weight * model.get_difference_loss_row_L2(True)
                if (configs.criterion.block_match_method == 'neighbor'):
                    diff_loss = configs.criterion.diff_loss_weight * model.get_difference_loss_row_L2_nei(True)

            loss = loss + configs.criterion.diff_loss_weight * diff_loss
        else:
            ortho_loss = torch.zeros(1)
        loss.backward()

        optimizer.step()

        step += 1

        if batch_idx % int(configs.run.log_interval) == 0:
            log = "Train Epoch: {} [{:7d}/{:7d} ({:3.0f}%)] Loss: {:.4f} Class Loss: {:.4f}".format(
                epoch,
                batch_idx * len(data),
                len(train_loader.dataset),
                100.0 * batch_idx / len(train_loader),
                loss.data.item(),
                class_loss.data.item(),
            )
            if configs.criterion.diff_loss_weight > 0:
                log += " Diff Loss: {:.4f}".format(diff_loss.item())
            lg.info(log)

    scheduler.step()
    accuracy = 100.0 * correct.float() / len(train_loader.dataset)
    lg.info(f"Train Accuracy: {correct}/{len(train_loader.dataset)} ({accuracy:.2f})%")


def validate(
    model: nn.Module,
    validation_loader: DataLoader,
    epoch: int,
    criterion: Criterion,
    loss_vector: Iterable,
    accuracy_vector: Iterable,
    device: torch.device,
) -> None:
    model.eval()
    val_loss, correct = 0, 0
    with torch.no_grad():
        for data, target in validation_loader:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(data)

            val_loss += criterion(output, target).data.item()
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).cpu().sum()

    val_loss /= len(validation_loader)
    loss_vector.append(val_loss)

    accuracy = 100.0 * correct.float() / len(validation_loader.dataset)
    accuracy_vector.append(accuracy)

    lg.info(
        "\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
            val_loss, correct, len(validation_loader.dataset), accuracy
        )
    )


def test(
    model: nn.Module,
    test_loader: DataLoader,
    epoch: int,
    criterion: Criterion,
    loss_vector: Iterable,
    accuracy_vector: Iterable,
    device: torch.device,
) -> None:
    model.eval()
    val_loss, correct = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(data)

            val_loss += criterion(output, target).data.item()
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).cpu().sum()

    val_loss /= len(test_loader)
    loss_vector.append(val_loss)

    accuracy = 100.0 * correct.float() / len(test_loader.dataset)
    accuracy_vector.append(accuracy)

    lg.info(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
            val_loss, correct, len(test_loader.dataset), accuracy
        )
    )

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE", help="config file")
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)

    if torch.cuda.is_available() and int(configs.run.use_cuda):
        torch.cuda.set_device(configs.run.gpu_id)
        device = torch.device("cuda:" + str(configs.run.gpu_id))
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
        torch.backends.cudnn.benchmark = False

    if int(configs.run.deterministic) == True:
        set_torch_deterministic()

    model = builder.make_model(
        device,
        int(configs.run.random_state) if int(configs.run.deterministic) else None,
        model_cfg=configs.model,
    )

    train_loader, validation_loader, test_loader = builder.make_dataloader()
    optimizer = builder.make_optimizer(
        [p for p in model.parameters() if p.requires_grad], configs.optimizer.name, configs.optimizer
    )
    scheduler = builder.make_scheduler(optimizer)
    criterion = builder.make_criterion().to(device)
    saver = BestKModelSaver(k=int(configs.checkpoint.save_best_model_k))

    lg.info(f"Number of parameters: {count_parameters(model)}")

    model_name = f"{configs.model.name}_{configs.dataset.img_height}x{configs.dataset.img_width}_ib-{configs.quantization.input_bit}_wb-{configs.quantization.weight_bit}_diff_loss_{configs.criterion.diff_loss_fn}_{configs.criterion.block_match_method}_{configs.criterion.diff_loss_weight}"

    checkpoint = f"./checkpoint/{configs.checkpoint.checkpoint_dir}/{model_name}"
    if len(configs.checkpoint.model_comment) > 0:
        checkpoint += "_" + configs.checkpoint.model_comment
    checkpoint += ".pt"

    lg.info(f"Current checkpoint: {checkpoint}")

    lossv, accv = [0], [0]
    epoch = 0
    try:
        
        lg.info(configs)
        if int(configs.checkpoint.resume):
            load_model(
                model,
                configs.checkpoint.restore_checkpoint,
                ignore_size_mismatch=int(configs.checkpoint.no_linear),
            )

            lg.info("Validate resumed model...")
            test(
                model,
                test_loader,
                0,
                criterion,
                [],
                [],
                device=device,
            )

        for epoch in range(1, int(configs.run.n_epochs) + 1):
            train(
                model,
                train_loader,
                optimizer,
                scheduler,
                epoch,
                criterion,
                device,
                teacher=None,
                soft_criterion=None,
            )
            if validation_loader is not None:
                lg.info(f"Validating model...")
                validate(
                    model,
                    validation_loader,
                    epoch,
                    criterion,
                    lossv,
                    accv,
                    device=device,
                )
                lg.info(f"Testing model...")
                test(
                    model,
                    test_loader,
                    epoch,
                    criterion,
                    [],
                    [],
                    device=device,
                )
            else:
                lg.info(f"Testing model...")
                test(
                    model,
                    test_loader,
                    epoch,
                    criterion,
                    lossv,
                    accv,
                    device=device,
                )
            saver.save_model(model, accv[-1], epoch=epoch, path=checkpoint, save_model=False, print_msg=True)
        save_model(model, checkpoint)
    except KeyboardInterrupt:
        lg.warning("Ctrl-C Stopped")

if __name__ == "__main__":
    main()