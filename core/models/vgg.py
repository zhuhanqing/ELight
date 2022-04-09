'''
Author: Hanqing Zhu(hqzhu@utexas.edu)
Date: 2022-04-07 10:38:18
LastEditTime: 2022-04-09 00:39:54
LastEditors: Hanqing Zhu(hqzhu@utexas.edu)
Description: 
FilePath: /projects/ELight/core/models/vgg.py
'''
'''Simplied versions of VGG8/11/13/16/19-bn in Pytorch.'''
import torch
import torch.nn as nn

from .layers import *
from .model_base import PCMBaseModel

__all__ = ['PCMVGG']

cfg = {
    'vgg8' : [64, 'M', 128, 'M', 256, 'M', 512, 'M', 512, 'M'],
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class PCMVGG(PCMBaseModel):
    def __init__(self, vgg_name,
        # unique params
        n_class=10,
        in_channel=3,
        block_size=32,
        in_bit=32,
        w_bit=32,
        mode='weight',
        input_quant_method='uniform_noise',
        weight_quant_method='log',
        quant_range='max',
        hasZero=True,
        p=0,
        assign=False,
        ### loss_flag
        loss_fn='l1',
        ### pcm param
        pcm_bit=4,
        pcm_l=0.128,
        device=torch.device("cuda")
    ):
        super(PCMVGG, self).__init__()
        # assign unique params
        self.bias = False
        self.n_class = n_class
        self.in_channel = in_channel
        self.in_bit = in_bit
        self.w_bit = w_bit
        self.mode = mode
        self.pcm_bit = pcm_bit
        self.pcm_l = pcm_l
        self.input_quant_method = input_quant_method
        self.weight_quant_method = weight_quant_method
        self.quant_range = quant_range
        self.hasZero = hasZero
        self.block_size=block_size
        self.device = device
        self.p = p
        self.loss_fn = loss_fn

        self.assign = assign

        self.convNum = 0

        self.features = self._make_layers(cfg[vgg_name])
        # self.classifier = nn.Linear(512, 10)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            PCMLinear(
                512,
                self.n_class,
                bias=self.bias,
                in_bit=self.in_bit,
                w_bit=self.w_bit,
                block_size=self.block_size,
                mode=self.mode,
                # quantization
                input_quant_method=self.input_quant_method,
                weight_quant_method=self.weight_quant_method,
                p=self.p,
                assign=self.assign,
                ### loss_flag
                loss_fn=self.loss_fn,
                ### PCM
                pcm_l=self.pcm_l,
                hasZero=self.hasZero,
                device=self.device
            )
        )

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, PCMConv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, PCMLinear):
                # nn.init.normal_(m.weight, 0, 0.1)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [PCMConv2d(
                        in_channels,
                        x,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=self.bias,
                        in_bit=self.in_bit,
                        w_bit= self.w_bit,
                        block_size=self.block_size,
                        mode=self.mode,
                        ### quantization
                        input_quant_method=self.input_quant_method,
                        weight_quant_method=self.weight_quant_method,
                        p=self.p,
                        assign=self.assign,
                        ### loss_flag
                        loss_fn=self.loss_fn,
                        ### pcm
                        pcm_l=self.pcm_l,
                        hasZero=self.hasZero,
                        device= self.device
                    ),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
                self.convNum += 1
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)