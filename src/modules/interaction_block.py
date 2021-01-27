import sympy as sym
import torch
import torch.nn as nn
import dgl
import dgl.function as fn

from modules.residual_layer import ResidualLayer
from modules.basis_utils import bessel_basis, real_sph_harm
from modules.envelope import Envelope

class InteractionBlock(nn.Module):
    def __init__(self,
                 emb_size,
                 num_radial,
                 num_spherical,
                 cutoff,
                 envelope_exponent,
                 num_bilinear,
                 num_before_skip,
                 num_after_skip,
                 sph_funcs,
                 activation=None):
        super(InteractionBlock, self).__init__()

        self.emb_size = emb_size
        self.num_radial = num_radial
        self.num_bilinear = num_bilinear
        self.activation = activation
        
        self.sph_funcs = sph_funcs

        # Transformations of Bessel and spherical basis representations
        self.dense_rbf = nn.Linear(num_radial, emb_size, bias=False)
        self.dense_sbf = nn.Linear(num_radial * num_spherical, num_bilinear, bias=False)
        # Dense transformations of input messages
        self.dense_ji = nn.Linear(emb_size, emb_size)
        self.dense_kj = nn.Linear(emb_size, emb_size)
        # Bilinear layer
        self.bilinear = nn.Bilinear(self.num_bilinear, self.emb_size, self.emb_size, bias=False)
        # Residual layers before skip connection
        self.layers_before_skip = nn.ModuleList([
            ResidualLayer(emb_size, activation=activation) for _ in range(num_before_skip)
        ])
        self.final_before_skip = nn.Linear(emb_size, emb_size)
        # Residual layers after skip connection
        self.layers_after_skip = nn.ModuleList([
            ResidualLayer(emb_size, activation=activation) for _ in range(num_after_skip)
        ])

        self.reset_params()
    
    def reset_params(self):
        nn.init.xavier_normal_(self.dense_rbf.weight)
        nn.init.xavier_normal_(self.dense_sbf.weight)
        nn.init.xavier_normal_(self.dense_ji.weight)
        nn.init.xavier_normal_(self.dense_kj.weight)
        bound = 2 / self.emb_size
        nn.init.uniform_(self.bilinear.weight, -bound, bound)

    @profile
    def edge_transfer(self, edges):
        # Transform via Bessel basis
        rbf = self.dense_rbf(edges.data['rbf'])
        # Initial transformation
        x_ji = self.dense_ji(edges.data['m'])
        x_kj = self.dense_kj(edges.data['m'])
        if self.activation is not None:
            x_ji = self.activation(x_ji)
            x_kj = self.activation(x_kj)

        # w: W * e_RBF \bigodot \sigma(W * m + b)
        return {'x_kj': x_kj * rbf, 'x_ji': x_ji}

    @profile
    def msg_func(self, edges):
        # Calculate angles k -> j -> i
        R1, R2 = edges.src['o'], edges.dst['o']
        x = torch.sum(R1 * R2, dim=-1)
        y = torch.cross(R1, R2)
        y = torch.norm(y, dim=-1)
        angle = torch.atan2(y, x)
        # Transform via angles
        cbf = [f(angle) for f in self.sph_funcs]
        cbf = torch.stack(cbf, dim=1)  # [None, 7]
        cbf = cbf.repeat_interleave(self.num_radial, dim=1)  # [None, 42]
        sbf = edges.src['rbf_env'] * cbf  # [None, 42]
        # Transform via spherical basis
        sbf = self.dense_sbf(sbf)

        # Apply bilinear layer to interactions and basis function activation
        # [None, 8] * [128, 8, 128] * [None, 128] -> [None, 128]
        x_kj = self.bilinear(sbf, edges.src['x_kj'])
        # sbf [None, 42]
        return {'x_kj': x_kj}

    @profile
    def forward(self, g, l_g):
        g.apply_edges(self.edge_transfer)
        
        # node means edge in original graph
        # edge means node in original graph
        # node: d, rbf, o
        # edge: R, Z, h
        for k, v in g.edata.items():
            l_g.ndata[k] = v

        l_g.update_all(self.msg_func, fn.sum('x_kj', 'm_update'))

        for k, v in l_g.ndata.items():
            g.edata[k] = v

        # Transformations before skip connection
        g.edata['m_update'] = g.edata['m_update'] + g.edata['x_ji']
        for layer in self.layers_before_skip:
            g.edata['m_update'] = layer(g.edata['m_update'])
        g.edata['m_update'] = self.final_before_skip(g.edata['m_update'])
        if self.activation is not None:
            g.edata['m_update'] = self.activation(g.edata['m_update'])

        # Skip connection
        g.edata['m'] = g.edata['m'] + g.edata['m_update']

        # Transformations after skip connection
        for layer in self.layers_after_skip:
            g.edata['m'] = layer(g.edata['m'])

        return g