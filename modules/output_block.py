import torch.nn as nn

from torch_scatter import scatter_add

class OutputBlock(nn.Module):
    def __init__(self,
                 emb_size,
                 num_radial,
                 num_dense,
                 num_targets=12,
                 activation=None):
        super(OutputBlock, self).__init__()

        self.activation = activation
        self.dense_rbf = nn.Linear(num_radial, emb_size, bias=False)
        self.dense_layers = nn.ModuleList([
            nn.Linear(emb_size, emb_size) for _ in range(num_dense)
        ])
        self.dense_final = nn.Linear(emb_size, num_targets, bias=False)

    def forward(self, inputs):
        x, rbf, idnb_i, n_atoms = inputs

        g = self.dense_rbf(rbf)
        x = g * x
        x = scatter_add(x, idnb_i, dim=0)
        for layer in self.dense_layers:
            x = layer(x)
            if self.activation is not None:
                x = self.activation(x)
        x = self.dense_final(x)
        return x