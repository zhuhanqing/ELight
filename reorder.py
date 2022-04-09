'''
Author: Hanqing Zhu(hqzhu@utexas.edu)
Date: 2022-04-07 10:35:27
LastEditTime: 2022-04-09 01:11:46
LastEditors: Hanqing Zhu(hqzhu@utexas.edu)
Description: Post-training reorder
FilePath: /projects/ELight/reorder.py
'''
import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from ops.utils import weight_to_quantized_weight_cpu
from collections import OrderedDict

import argparse

class model_analyzer:
    def __init__(self, pcm_l):
        self.pcm_l = pcm_l

    def load_model(self, checkpoint_path, bits, diff_loss_weight):
        self.checkpoint_path = checkpoint_path
        self.bits = bits
        self.diff_loss_weight = diff_loss_weight
        
        # obtain weights from state_dict
        raw_data = torch.load(checkpoint_path)
        if(isinstance(raw_data, OrderedDict) and "state_dict" not in raw_data):
            self.model_dict = raw_data
        else:
            self.model_dict = raw_data["state_dict"]
        print('[I] Successfully load data from {}'.format(checkpoint_path))
        self.dict_name = list(self.model_dict) # obtain its keys

        self.weight2level = weight_to_quantized_weight_cpu(bits, 1- self.pcm_l , True, True, 2**bits - 1, False) # combine
        self.weight2level_sep = weight_to_quantized_weight_cpu(bits, 1- self.pcm_l , True, True, 2**bits - 1, True) # sep

        if "vgg8" in checkpoint_path:
            self.model_name = f'vgg8-cifar10-ib-{self.bits}-wb-{bits}'
            self.layer_list = [0, 9, 18, 27, 36, 45]
        
        if 'l1' in checkpoint_path:
            loss_fn = 'l1'
        else:
            loss_fn = 'l2'
        if 'neighbor' in checkpoint_path:
            block_match_method = 'neighbor'
        elif 'meanG' in checkpoint_path:
            block_match_method = 'meanG'
        else:
            block_match_method = 'meanR'

        self.model_comment = f"diff_loss_{loss_fn}_{block_match_method}_{diff_loss_weight}"

        print(f"Current model is {self.model_name}_{self.model_comment}")

    def set_model_comment(self, comment):
        self.model_comment = f'{self.model_comment}-{comment}'

    def analyze_model(self, isReorder):
        '''analyze given model
            isReorder: whether we use reorder heursitic: sort
        '''
        print("Start analyze model data : {}".format(self.model_comment))
        self.isReorder = isReorder

        # initialize output data container
        self.wt_per_layer = OrderedDict() # p*q blocks
        self.sum_wt_per_layer = OrderedDict() # sum WT per layer
        self.sum_wt_per_layer_list = [] # list to store the sum
        self.sum_wt_per_layer_l2h = OrderedDict() # sum WT per layer from low absorption to high (a-c)
        self.sum_wt_per_layer_l2h_list = [] # list to store the sum
        self.sum_wt_per_layer_h2l = OrderedDict() # sum WT per layer from high absorption to low (c-a)
        self.sum_wt_per_layer_h2l_list = [] # list to store the sum
        self.max_wt_latency_per_layer = OrderedDict() # max programing latency for one layer
        # we assume for p PTC we parallel programming
        # also for postive and negative PTC we also program in parallel
        self.max_wt_latency_per_layer_list = []
        self.energy_cost_per_layer = OrderedDict()
        self.energy_cost_per_layer_list = []
        
        self.max_wt_per_layer_real = OrderedDict()
        self.max_wt_per_layer_real_list = []

        for idx, layer_idx in enumerate(self.layer_list):
            if 'vgg8' in self.model_name:
                layer_name = ("conv" + str(idx)) if idx <= 4 else "fc" + str(idx - 5)
            weight = self.model_dict[self.dict_name[layer_idx]]

            # check size
            print(layer_name)
            
            p, q, k, k = weight.size()
            # reshape weight_data
            weight_data = weight.view(p*q, -1)

            _, levels_data = self.weight2level.forward(weight_data) # [p*q , k*k]
            _, levels_data_sep = self.weight2level_sep.forward(weight_data) # obtain sep levels data [p*q , k*k*2]

            levels_data = levels_data.mul(2**self.bits - 1).view(p, q , k, k)
            levels_data_sep_pn = torch.chunk(levels_data_sep, 2, dim=1)
            levels_data_sep_p = levels_data_sep_pn[0].view(p, q, k, k).mul(2**self.bits - 1)
            levels_data_sep_n = levels_data_sep_pn[1].view(p, q, k, k).mul(2**self.bits - 1)
            if ('vgg8' in self.model_name) and ('fc' in layer_name):
                # last layer dimension is [512, 10], we cannot compute the results of padding part since we use a 64 64 block
                levels_data[:, :, 10:, :] = float('inf')
                levels_data_sep_p[:, :, 10:, :] = float('inf')
                levels_data_sep_n[:, :, 10:, :] = 0

                existInfinite = True
            else:
                existInfinite = False
            # reorder or not
            if isReorder:
                print('Reordering...')
                levels_data_ordered, levels_data_sep_p_ordered, levels_data_sep_n_ordered = self.reorder(levels_data, levels_data_sep_p, levels_data_sep_n)
            else:
                print('No reorder...')
                # directly clone
                levels_data_ordered = levels_data.clone()
                levels_data_sep_p_ordered = levels_data_sep_p.clone()
                levels_data_sep_n_ordered = levels_data_sep_n.clone()

            wt_per_row, max_wt_per_row, wt_sum, energyCost_sum, max_latency, max_wt = self.analyze_wt(levels_data_ordered, levels_data_sep_p_ordered, levels_data_sep_n_ordered, existInfinite)
            # print(f"check wt_sum at {idx}: {wt_sum}")
            tmp_mean = []
            tmp_std  = []
            tmp_max  = []

            # save wt_per_row
            for i in range(len(wt_per_row)):
                name_row = layer_name + '_row' + str(i)
                self.wt_per_layer[name_row] = np.array(wt_per_row[i])
                tmp_mean.append(np.mean(self.wt_per_layer[name_row]))
                tmp_std.append(np.std(self.wt_per_layer[name_row]))
                # max time
                tmp_max.append(np.max(max_wt_per_row[i]))
            
            self.sum_wt_per_layer_list.append(wt_sum)
            self.max_wt_latency_per_layer_list.append(max_latency)
            self.energy_cost_per_layer_list.append(energyCost_sum)
            self.max_wt_per_layer_real_list.append(max_wt)

        name_sum  = 'sum by layers'
        name_max_latency = 'max_latency'
        name_energy_cost = 'energy_cost'
        name_max_wt = 'max_wt'

        self.sum_wt_per_layer[name_sum] = np.array(self.sum_wt_per_layer_list)
        self.max_wt_latency_per_layer[name_max_latency] = np.array(self.max_wt_latency_per_layer_list)
        self.energy_cost_per_layer[name_energy_cost] = np.array(self.energy_cost_per_layer_list)
        self.max_wt_per_layer_real[name_max_wt] = np.array(self.max_wt_per_layer_real_list)
        # merge dicts
        dictsMerge = self.wt_per_layer.copy()
        dictsMerge.update(self.sum_wt_per_layer)
        dictsMerge.update(self.max_wt_latency_per_layer)
        dictsMerge.update(self.energy_cost_per_layer)
        dictsMerge.update(self.max_wt_per_layer_real)

        # output
        if isReorder:
            reorder_comment = 'reorder'
        else:
            reorder_comment = 'noreorder'
        file_name = f'{self.model_comment}-{reorder_comment}'
        file_path = f"./logs/{self.model_name}/{file_name}.csv"

        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))

        save_data = pd.DataFrame.from_dict(dictsMerge, orient='index')
        save_data.to_csv(file_path)
        print('[I] File is saved into {}'.format(file_path))

    def analyze_wt(self, levels_data, levels_data_sep_p, levels_data_sep_n, isInfinite):
        # timeData
        timeLow2High = 0.5 # 0.5us
        timeHigh2Low = 20 # 1us: 20
        # energyData: P= IVt = V^2 t /R
        energyLow2High = 9
        energyHigh2Low = 40

        p, q, k, k = levels_data.size()
        levels = levels_data.view(p*q, -1)
        levels_pos = levels_data_sep_p.view(p*q, -1)
        levels_neg = levels_data_sep_n.view(p*q, -1)

        # obtain ref values
        # we assume when mapping weight blocks, the previous data is cleaned at first time: -> set to 0
        indices = torch.arange(0, p*q, 1).long()
        indices = indices - 1

        # init ref data
        levels_ref = levels.detach().clone()
        levels_pos_ref = levels_pos.detach().clone()
        levels_neg_ref = levels_neg.detach().clone()
        base = torch.unsqueeze(levels[0], 0)
        for i in range(p):
            base_0 = torch.zeros_like(base)
            levels_ref = torch.cat((levels_ref, base_0), 0)
            levels_pos_ref = torch.cat((levels_pos_ref, base_0), 0)
            levels_neg_ref = torch.cat((levels_neg_ref, base_0), 0)

            # update indices
            indices[i*q] = p*q + i
        
        levels_ref = levels_ref[indices]
        levels_pos_ref = levels_pos_ref[indices]
        levels_neg_ref = levels_neg_ref[indices]
        # pre-set inf to 0 to avoid nan (inf - inf = nan)
        if isInfinite:
            levels_ref[levels_ref == float("inf")] = 0
            levels_pos_ref[levels_pos_ref == float("inf")] = 0
            levels_neg_ref[levels_neg_ref == float("inf")] = 0
            
        tmp_diff = F.l1_loss(levels, levels_ref, reduction='none').detach() # p*q, k*k
        tmp_diff_pos = levels_pos - levels_pos_ref
        tmp_diff_neg = levels_neg - levels_neg_ref

        if isInfinite:
            tmp_diff[tmp_diff == float("inf")] = 0
            tmp_diff_pos[tmp_diff_pos == float("inf")] = 0
            tmp_diff_neg[tmp_diff_neg == float("inf")] = 0

        tmp_diff_block = tmp_diff.sum(dim=1) # p*q
        
        mask_pos_pos = tmp_diff_pos > 0
        mask_pos_neg = tmp_diff_pos < 0
        mask_neg_pos = tmp_diff_neg > 0
        mask_neg_neg = tmp_diff_neg < 0

        # latency
        tmp_diff_pos_latency = tmp_diff_pos.detach().clone()
        tmp_diff_neg_latency = tmp_diff_neg.detach().clone()

        tmp_diff_pos_latency[mask_pos_pos] = tmp_diff_pos[mask_pos_pos] * timeLow2High
        tmp_diff_pos_latency[mask_pos_neg] = tmp_diff_pos[mask_pos_neg] * timeHigh2Low
        tmp_diff_neg_latency[mask_neg_pos] = tmp_diff_pos[mask_neg_pos] * timeHigh2Low
        tmp_diff_neg_latency[mask_neg_neg] = tmp_diff_neg[mask_neg_neg] * timeLow2High

        # we can obtain the max programming latecny on one cells
        mask_max_latency = torch.abs(tmp_diff_pos_latency) > torch.abs(tmp_diff_neg_latency)
        tmp_diff_max_time = torch.abs(tmp_diff_pos_latency) # p*q, k*k
        tmp_diff_max_time[~mask_max_latency] = tmp_diff_neg_latency.abs()[~mask_max_latency]
        # print(tmp_diff_max_time)
        # energy
        tmp_diff_pos_energy = tmp_diff_pos.detach().clone()
        tmp_diff_neg_energy = tmp_diff_neg.detach().clone()

        tmp_diff_pos_energy[mask_pos_pos] = tmp_diff_pos[mask_pos_pos] * energyLow2High
        tmp_diff_pos_energy[mask_pos_neg] = tmp_diff_pos[mask_pos_neg] * energyHigh2Low
        tmp_diff_neg_energy[mask_neg_pos] = tmp_diff_pos[mask_neg_pos] * energyLow2High
        tmp_diff_neg_energy[mask_neg_neg] = tmp_diff_neg[mask_neg_neg] * energyHigh2Low

        tmp_diff_energy = torch.abs(tmp_diff_pos_energy) + torch.abs(tmp_diff_neg_energy) # p*q, k*k

        # chunk
        wt_chunk = torch.chunk(tmp_diff_block, p, dim=0) # p*q
        wt_diff_chunk = torch.chunk(tmp_diff, p, dim=0) # p*q, k*k
        # print(tmp_diff_block)
        wt_per_row = []
        max_wt_per_row = []
        wt_sum = torch.sum(tmp_diff_block).item()
        energyCost_sum = torch.sum(tmp_diff_energy).item()

        for i in range(p):
            wt_per_row.append(wt_chunk[i].cpu().numpy().tolist()) # q
            max_wt_per_row.append(torch.max(wt_diff_chunk[i].sum(dim=0)[0:k*k]).item()) # 1
        tmp_max_latency, _ = torch.max(tmp_diff_max_time, -1) #p*q
        max_latency_each_map_step, _ = torch.max(tmp_max_latency.view(p, q), 0) # q
        max_latency = max_latency_each_map_step.sum(dim=0).item()
        max_wt = max(max_wt_per_row)
        return wt_per_row, max_wt_per_row, wt_sum, energyCost_sum, max_latency, max_wt

    def reorder(self, x, x_sep_pos, x_sep_neg):
        ## sep levels data should follow the same reorder indices with x
        # x: p, q, k, k
        x_ordered, indices = torch.sort(x, dim=1)
        x_sep_pos_ordered = torch.gather(x_sep_pos, 1, indices)
        x_sep_neg_ordered = torch.gather(x_sep_neg, 1, indices)

        return x_ordered, x_sep_pos_ordered, x_sep_neg_ordered

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint", type=str, default="./checkpoint/cifar/vgg/model.pt", help="checkpoint file")
    parser.add_argument("-d", "--diff_loss_weight", type=float, default=0, help="diff loss weight")
    parser.add_argument("-b", "--bits", type=float, default=4, help="weight bits")
    args, opts = parser.parse_known_args()

    reorder = model_analyzer(0.128)

    reorder.load_model(args.checkpoint, args.bits, args.diff_loss_weight)
    reorder.analyze_model(True)
    reorder.analyze_model(False)