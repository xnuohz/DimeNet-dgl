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
    
        self.reset_params()
    
    def reset_params(self):
        nn.init.xavier_normal_(self.dense_1.weight)
        nn.init.xavier_normal_(self.dense_2.weight)

    def forward(self, inputs):
        x = self.dense_1(inputs)
        if self.activation is not None:
            x = self.activation(x)
        x = self.dense_2(x)
        if self.activation is not None:
            x = self.activation(x)
        return inputs + x