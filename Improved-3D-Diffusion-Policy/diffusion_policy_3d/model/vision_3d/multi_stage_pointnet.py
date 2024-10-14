import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from termcolor import cprint

def meanpool(x, dim=-1, keepdim=False):
    out = x.mean(dim=dim, keepdim=keepdim)
    return out

def maxpool(x, dim=-1, keepdim=False):
    out = x.max(dim=dim, keepdim=keepdim).values
    return out

class MultiStagePointNetEncoder(nn.Module):
    def __init__(self, h_dim=128, out_channels=128, num_layers=4, **kwargs):
        super().__init__()

        self.h_dim = h_dim
        self.out_channels = out_channels
        self.num_layers = num_layers

        self.act = nn.LeakyReLU(negative_slope=0.0, inplace=False)

        self.conv_in = nn.Conv1d(3, h_dim, kernel_size=1)
        self.layers, self.global_layers = nn.ModuleList(), nn.ModuleList()
        for i in range(self.num_layers):
            self.layers.append(nn.Conv1d(h_dim, h_dim, kernel_size=1))
            self.global_layers.append(nn.Conv1d(h_dim * 2, h_dim, kernel_size=1))
        self.conv_out = nn.Conv1d(h_dim * self.num_layers, out_channels, kernel_size=1)

    def forward(self, x):
        x = x.transpose(1, 2) # [B, N, 3] --> [B, 3, N]
        y = self.act(self.conv_in(x))
        feat_list = []
        for i in range(self.num_layers):
            y = self.act(self.layers[i](y))
            y_global = y.max(-1, keepdim=True).values
            y = torch.cat([y, y_global.expand_as(y)], dim=1)
            y = self.act(self.global_layers[i](y))
            feat_list.append(y)
        x = torch.cat(feat_list, dim=1)
        x = self.conv_out(x)

        x_global = x.max(-1).values

        return x_global

        


   