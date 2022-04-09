'''
Author: Hanqing Zhu(hqzhu@utexas.edu)
Date: 2022-04-07 10:38:08
LastEditTime: 2022-04-08 23:49:11
LastEditors: Hanqing Zhu(hqzhu@utexas.edu)
Description: Base model
FilePath: /projects/ELight/core/models/model_base.py
'''
import torch.nn as nn
from typing import Callable, Dict, List, Optional, Tuple, Union

# import layers
from .layers import *

class PCMBaseModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # loss func
    def get_difference_loss_global_L1(self, loss_flag):
        # compute loss on average block
        loss = 0
        for m in self.modules():
            if isinstance(m, PCMConv2d):
                loss = loss + m.get_difference_loss_global_L1(loss_flag)
            if isinstance(m, PCMLinear):
                loss = loss + m.get_difference_loss_global_L1(loss_flag)

        return loss

    def get_difference_loss_row_L1(self, loss_flag):
        # compute loss on average block by rows
        loss = 0
        for m in self.modules():
            if isinstance(m, PCMConv2d):
                loss = loss + m.get_difference_loss_row_L1(loss_flag)
            if isinstance(m, PCMLinear):
                loss = loss + m.get_difference_loss_row_L1(loss_flag)

        return loss

    def get_difference_loss_global_L2(self, loss_flag):
        # compute loss on average block by all
        loss = 0
        for m in self.modules():
            if isinstance(m, PCMConv2d):
                loss = loss + m.get_difference_loss_global_L2(loss_flag)
            if isinstance(m, PCMLinear):
                loss = loss + m.get_difference_loss_global_L2(loss_flag)

        return loss

    def get_difference_loss_row_L2(self, loss_flag):
        # compute loss on average block by rows
        loss = 0
        for m in self.modules():
            if isinstance(m, PCMConv2d):
                loss = loss + m.get_difference_loss_row_L2(loss_flag)
            if isinstance(m, PCMLinear):
                loss = loss + m.get_difference_loss_row_L2(loss_flag)

        return loss

    def get_difference_loss_row_L2_nei(self, loss_flag):
        # compute loss on average block by rows
        loss = 0
        for m in self.modules():
            if isinstance(m, PCMConv2d):
                loss = loss + m.get_difference_loss_row_L2_nei(loss_flag)
            if isinstance(m, PCMLinear):
                loss = loss + m.get_difference_loss_row_L2_nei(loss_flag)

        return loss

    def get_programming_levels_global_real(self, loss_flag):
        # compute loss on average block
        loss = 0
        for m in self.modules():
            if isinstance(m, PCMConv2d):
                loss = loss + m.get_programming_levels_global_real(loss_flag)
            if isinstance(m, PCMLinear):
                loss = loss + m.get_programming_levels_global_real(loss_flag)

        return loss
