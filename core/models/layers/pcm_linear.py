'''
Author: Hanqing Zhu(hqzhu@utexas.edu)
Date: 2022-04-07 10:37:05
LastEditTime: 2022-04-08 23:45:23
LastEditors: Hanqing Zhu(hqzhu@utexas.edu)
Description: 
FilePath: /projects/ELight/core/models/layers/pcm_linear.py
'''
import logging

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter, init

import math # some math operations

# import needed func from pyutils @https://github.com/JeremieMelo/pyutility
from pyutils.quantize import input_quantize_fn
from pyutils.torch_train import set_torch_deterministic
from pyutils.general import print_stat
from ops.utils import weight_quantize_fn_log, weight_to_quantized_weight, weight_to_quantized_weight_cpu

__all__ = ["PCMLinear"]

class PCMLinear(nn.Module):
    """
    description: PCM linear layer with weight quantization and mapping
    """
    def __init__(self,
        ## normal params 
        in_channel,
        out_channel,
        bias=False,
        w_bit=16,
        in_bit=16,
        block_size=16,
        mode='weight',
        ### quantization
        input_quant_method="uniform_noise",
         # quant_noise
        p = 0,
        weight_quant_method='log',
        ## loss_flag
        loss_fn='l1',
        ## pcm params
        pcm_l=0.128,
        assign=False,
        hasZero=True,
        device=torch.device("cuda")
    ):
        super(PCMLinear, self).__init__()

        # param init
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.w_bit = w_bit
        self.in_bit = in_bit
        self.pcm_l = pcm_l
        self.device = device
        self.p = p

       # PTC param
        self.mode = mode
        assert mode in {"weight", "block"}, logging.error(f"Mode not supported. Expected one from (weight, block) but got {mode}.")
        self.block_size = block_size
        self.block_mul_flag = True if (mode == 'block') else False

        self.input_quant_method = input_quant_method
        self.weight_quant_method = weight_quant_method
        self.hasZero = hasZero

        self.assign = assign
        self.stuck_fault = False
        self.tolerate_stuck_fault = False

        ## allocate the trainable parameters
        self.weight = None
        self.build_parameters()
        if(bias):
            self.bias = Parameter(torch.Tensor(self.out_channel).to(self.device))
        else:
            self.register_parameter('bias', None)

        ## Initialize quantization tools
        if self.in_bit < 16:
                self.input_quantizer = input_quantize_fn(self.in_bit, alg='normal', device=self.device, quant_ratio=self.p)
                
        if self.w_bit < 16:
            if(self.weight_quant_method == "log"):
                self.weight_quantizer = weight_quantize_fn_log(self.w_bit, power_base=1-self.pcm_l, hasZero=self.hasZero, power=True, assign=self.assign, device=self.device)
            else:
                assert NotImplementedError
        else:
            pass
        
        assign_zero_value = 2**self.w_bit - 1

        if self.assign and (self.w_bit < 16):
            self.assign_converter = weight_to_quantized_weight(self.w_bit, 1-self.pcm_l, True, self.assign, assign_zero_value, loss_fn)
            self.real_assign_converter = weight_to_quantized_weight_cpu(self.w_bit, 1-self.pcm_l, True, self.assign, assign_zero_value)

        # defualt settings
        self.disable_fast_forward()

    def enable_fast_forward(self):
        self.fast_forward_flag = True

    def disable_fast_forward(self):
        self.fast_forward_flag = False

    def build_parameters(self):
        if self.mode in {"weight"}:
            self.weight = torch.Tensor(self.out_channel, self.in_channel).to(self.device).float()
            self.weight = Parameter(self.weight)
        elif (self.mode == "block"):
            self.weight = Parameter(torch.Tensor((self.out_channel + self.block_size - 1) // self.block_size, (self.in_channel + self.block_size - 1) // self.block_size, self.block_size, self.block_size))
        else:
            self.weight = Parameter(torch.Tensor(self.out_channel, self.in_channel).to(self.device).float())

    def reset_parameters(self):
        if (self.mode in {"weight", "block"}):
            init.kaiming_normal_(self.weight.data, mode="fan_out", nonlinearity="relu")
        else:
            init.kaiming_normal_(self.weight.data, mode="fan_out", nonlinearity="relu")

        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, 0, 0)

    def build_weight(self):
        if self.mode in {"weight"}:
            if(self.w_bit < 16):
                weight = self.weight_quantizer(self.weight)
            else:
                weight = self.weight
            weight = weight.view(self.out_channel, -1)[:, :self.in_channel]
        elif (self.mode == "block"):
            if(self.w_bit < 16):
                weight = self.weight_quantizer(self.weight)
            else:
                weight = self.weight

            # reshape to out_channel * in_channel from p, q, k, k
            p, q, k, k = weight.size()
            weight = weight.permute([0, 2, 1, 3]).contiguous().view(p*k, q*k)[:self.out_channel, :self.in_channel].view(self.out_channel, self.in_channel).contiguous()
            
        return weight

    def forward(self, x):
        if(self.in_bit < 16):
            if ('noise' in self.input_quant_method):
                p = self.p if self.training else 1
                self.input_quantizer.set_quant_ratio(p)
                x = self.input_quantizer(x)
            else:
                x = self.input_quantizer(x)

        if (not self.fast_forward_flag or self.weight is None):
            weight = self.build_weight()
        else:
            weight = self.weight.view(self.out_channel, -1)[:, :self.in_channel]

        out = F.linear(x, weight, self.bias)
        
        return out

    def get_difference_loss_global_L1(self, loss_flag):
        '''
        Obtain the difference loss between blocks and ref block in a global manner
        '''
        loss = 0
        if (self.block_mul_flag == True):
            p, q, k, k = self.weight.size()
            weight = self.weight.view(p*q, -1)
            weight = self.assign_converter(weight) # transfer weight to transmission levels
            base = torch.mean(weight, dim=0, keepdim=True).detach() # get one global reference block
            ref = base.repeat(p*q, 1)

            if loss_flag:
                loss = F.l1_loss(weight, ref, reduction='mean')
            else:
                tmp = F.l1_loss(weight, ref, reduction='none').detach().sum(dim=1).div(k*k)
                self.block_full_differences = tmp.cpu().numpy().tolist()
        else:
            loss = 0
        
        return loss

    def get_difference_loss_global_L2(self, loss_flag):
        ''' L2 loss
        Obtain the difference loss between blocks and ref block in a global manner
        '''
        loss = 0
        if (self.block_mul_flag == True):
            p, q, k, k = self.weight.size()
            weight = self.weight.view(p*q, -1)
            weight = self.assign_converter(weight)
            base = torch.mean(weight, dim=0, keepdim=True).detach() # get one global reference block
            ref = base.repeat(p*q, 1)

            if loss_flag:
                loss = F.mse_loss(weight, ref, reduction='mean')
            else:
                tmp = F.mse_loss(weight, ref, reduction='none').detach().sum(dim=1).div(k*k)
                self.block_full_differences_real = tmp.cpu().numpy().tolist()
        else:
            loss = 0
        
        return loss


    def get_difference_loss_row_L1(self, loss_flag):
        '''
        Obtain the difference loss between blocks and ref block in a row-wise manner
        '''
        loss = 0
        
        if (self.block_mul_flag == True):
            p, q, k, k = self.weight.size()
            weight = self.weight.view(p*q, -1)
            weight = self.assign_converter(weight)

            for i in range(p):
                base = torch.mean(weight[i*q:i*q+q], dim=0, keepdim=True).detach() # get one reference block for each row
                if i == 0:
                    ref = base.repeat(q, 1)
                else:
                    ref = torch.cat((ref, base.repeat(q, 1)), 0)
                    
            if loss_flag:
                loss = F.l1_loss(weight, ref, reduction='mean')
            else:
                tmp = F.l1_loss(weight, ref, reduction='none').detach().sum(dim=1).div(k*k)
                tmp_chunk = torch.chunk(tmp, p, dim=0)
                self.block_avr_differences_row = []
                for i in range(p):
                    self.block_avr_differences_row.append(tmp_chunk[i].cpu().numpy().tolist())
        else:
            loss = 0
        
        return loss

    
    def get_difference_loss_row_L2(self, loss_flag):
        '''
        Obtain the difference loss between blocks and ref block in a row-wise manner
        '''
        loss = 0
        
        if (self.block_mul_flag == True):
            p, q, k, k = self.weight.size()
            weight = self.weight.view(p*q, -1)
            weight = self.assign_converter(weight)
            for i in range(p):
                base = torch.mean(weight[i*q:i*q+q], dim=0, keepdim=True).detach() # get one reference block for each row
                if i == 0:
                    ref = base.repeat(q, 1)
                else:
                    ref = torch.cat((ref, base.repeat(q, 1)), 0)
            if loss_flag:
                loss = F.mse_loss(weight, ref, reduction='mean')
            else:
                tmp = F.mse_loss(weight, ref, reduction='none').detach().sum(dim=1).div(k*k)
                tmp_chunk = torch.chunk(tmp, p, dim=0)
                self.block_avr_differences_row_real = []
                for i in range(p):
                    self.block_avr_differences_row_real.append(tmp_chunk[i].cpu().numpy().tolist())
        else:
            loss = 0
        
        return loss   

    def get_difference_loss_nei_L2(self, loss_flag):
        '''
        Obtain the difference loss between neighbouring blocks
        '''
        loss = 0
        
        if (self.block_mul_flag == True):
            p, q, k, k = self.weight.size()
            weight = self.weight.view(p*q, -1)
            weight = self.assign_converter(weight)

            indices = torch.arange(0, p*q, 1).long()
            indices = indices - 1
            weight_ref = weight.detach().clone()

            for i in range(p):
                base = torch.mean(weight[i*q:i*q+q], dim=0, keepdim=True).detach()
                indices[i*q] = p*q + i
                weight_ref = torch.cat((weight_ref, base), 0)

            weight_ref = weight_ref[indices]

            if loss_flag:
                loss = F.mse_loss(weight, weight_ref, reduction='mean')
            else:
                tmp = F.mse_loss(weight, weight_ref, reduction='none').detach().sum(dim=1).div(k*k)
                tmp_chunk = torch.chunk(tmp, p, dim=0)
                self.block_avr_differences_row_real = []
                for i in range(p):
                    self.block_avr_differences_row_real.append(tmp_chunk[i].cpu().numpy().tolist())
        else:
            loss = 0
        
        return loss

    def get_programming_levels_v4_real(self, loss_flag):
        '''
        Compute the total number of write operations
            Use the avearge block for each row as the initialization block
        '''
        loss = 0
        
        if (self.block_mul_flag == True):
            p, q, k, k = self.weight.size()
            weight = self.weight.view(p*q, -1)
            _, weight = self.real_assign_converter.forward(weight)

            indices = torch.arange(0, p*q, 1).long()
            indices = indices - 1
            weight_ref = weight.detach().clone()
            for i in range(p):
                base = torch.mean(weight[i*q:i*q+q], dim=0, keepdim=True)
                indices[i*q] = p*q + i
                weight_ref = torch.cat((weight_ref, base), 0)

            weight_ref = weight_ref[indices]
                    
            if loss_flag:
                loss = F.l1_loss(weight, weight_ref, reduction='mean')
            else:
                tmp = F.l1_loss(weight, weight_ref, reduction='none').detach().sum(dim=1).div(k*k)
                tmp_chunk = torch.chunk(tmp, p, dim=0)
                self.programming_levels_avr_row_real = []
                for i in range(p):
                    self.programming_levels_avr_row_real.append(tmp_chunk[i].cpu().numpy().tolist())
        else:
            loss = 0
        
        return loss

if __name__ == "__main__":
    device = "cuda"
    layer = PCMLinear(128, 64, w_bit=4, in_bit=32, mode='block', block_size=32, assign=True, device=device)
    layer.reset_parameters()
    input = torch.randn(20, 128)
    output = layer(input)
    print_stat(output)
