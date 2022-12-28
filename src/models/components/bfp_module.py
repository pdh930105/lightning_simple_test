import torch.nn as nn
import torch
from qtorch.quant import Quantizer, quantizer
from qtorch.optim import OptimLP
from torch.optim import SGD
from qtorch import FloatingPoint, BlockFloatingPoint


class BFPConv2d(nn.Module):
    def __init__(self, m, e_bit, m_bit):
        super().__init__()
        self.conv = m
        self.e_bit= e_bit
        self.m_bit = m_bit
    
    def forward(self, x):
        x = self.conv(x)
        return x
    
def monkey_patch(old_net):
    for old_n, old_m in old_net.named_children():
        if isinstance(old_m, nn.Conv2d):
            setattr(old_net, old_n, BFPConv2d(old_m, 4, 4))
        elif isinstance(old_n, nn.Linear):
            print("123")
        else:
            monkey_patch(old_m)