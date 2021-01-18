import torch.nn as nn

class ResidualLayer(nn.Module):
    def __init__(self,
                 units,
                 activation=None,
                 use_bias=True):
        super(ResidualLayer, self).__init__()

        self.activation = activation
        self.dense_1 = nn.Linear(units, units, bias=use_bias)
        self.dense_2 = nn.Linear(units, units, bias=use_bias)
    
    def forward(self, inputs):
        x = self.dense_1(inputs)
        if self.activation is not None:
            x = self.activation(x)
        x = self.dense_2(x)
        if self.activation is not None:
            x = self.activation(x)
        return inputs + x