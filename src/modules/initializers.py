import torch
import torch.nn as nn

def GlorotOrthogonal(tensor, scale=2.0):
    assert len(tensor.size()) == 2
    nn.init.orthogonal_(tensor)
    tensor *= torch.sqrt(scale / ((tensor.size()[0] + tensor.size()[1]) * torch.var(tensor)))