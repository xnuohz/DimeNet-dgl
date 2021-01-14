import torch.nn as nn
import torch

from modules.residual_layer import ResidualLayer
from torch_scatter import scatter_add

class InteractionBlock(nn.Module):
    def __init__(self,
                 emb_size,
                 num_radial,
                 num_spherical,
                 num_bilinear,
                 num_before_skip,
                 num_after_skip,
                 activation=None):
        super(InteractionBlock, self).__init__()
        
        self.emb_size = emb_size
        self.num_bilinear = num_bilinear
        self.activation = activation

        # Transformations of Bessel and spherical basis representations
        self.dense_rbf = nn.Linear(num_radial, emb_size, bias=False)
        self.dense_sbf = nn.Linear(num_radial * num_spherical, emb_size, bias=False)

        # Dense transformations of input messages
        self.dense_ji = nn.Linear(emb_size, emb_size)
        self.dense_kj = nn.Linear(emb_size, emb_size)

        # Bi-linear
        bilin_initializer = torch.empty((emb_size, num_bilinear, emb_size)).normal_(mean=0, std=2 / emb_size)
        self.W_bilin = nn.Parameter(bilin_initializer)

        # Residual layers before skip connection
        self.layers_before_skip = nn.ModuleList([
            ResidualLayer(emb_size,
                          activation=activation,
                          use_bias=True) for _ in range(num_before_skip)
        ])

        self.final_before_skip = nn.Linear(emb_size, emb_size)

        # Residual layers after skip connection
        self.layers_after_skip = nn.ModuleList([
            ResidualLayer(emb_size,
                          activation=activation,
                          use_bias=True) for _ in range(num_after_skip)
        ])

    def forward(self, inputs):
        x, rbf, sbf, id_expand_kj, id_reduce_ji = inputs

        # Initial transformation
        x_ji = self.dense_ji(x)
        x_kj = self.dense_kj(x)

        # Transform via Bessel basis
        g = self.dense_rbf(rbf)
        x_kj = x_kj * g

        # Transform via spherical basis
        sbf = self.dense_sbf(sbf)
        x_kj = x_kj[id_expand_kj]

        # Apply bilinear layer to interactions and basis function activation
        x_kj = torch.einsum("wj,wl,ijl->wi", sbf, x_kj, self.W_bilin)
        x_kj = scatter_add(x_kj, id_reduce_ji, dim=0)  # sum over messages

        # Transformations before skip connection
        x2 = x_ji + x_kj
        for layer in self.layers_before_skip:
            x2 = layer(x2)
            if self.activation is not None:
                x2 = self.activation(x2)
        x2 = self.final_before_skip(x2)
        if self.activation is not None:
            x2 = self.activation(x2)

        # Skip connection
        x = x + x2

        # Transformations after skip connection
        for layer in self.layers_after_skip:
            x = layer(x)
            if self.activation is not None:
                x = self.activation(x)
        return x