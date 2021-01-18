import torch.nn as nn
import torch
import dgl
import sympy as sym

from modules.residual_layer import ResidualLayer
from modules.basis_utils import bessel_basis, real_sph_harm
from modules.envelope import Envelope
from torch_scatter import scatter_add

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

        self.dense_rbf = nn.Linear(num_radial, emb_size, bias=False)
        self.dense_sbf = nn.Linear(num_radial * num_spherical, num_bilinear, bias=False)
        self.dense_m = nn.Linear(emb_size, emb_size)

        bilin_initializer = torch.empty((self.emb_size, self.num_bilinear, self.emb_size)).normal_(mean=0, std=2 / emb_size)
        self.W_bilin = nn.Parameter(bilin_initializer)

    def edge_transfer(self, edges):
        # from rbf layer
        rbf = self.dense_rbf(edges.data['rbf'])
        m = self.dense_m(edges.data['m'])
        if self.activation is not None:
            m = self.activation(m)

        # w: W * e_RBF \bigodot \sigma(W * m + b)
        return {'w': rbf * m, 'rbf_env': torch.ones([20, 42])}

    def msg_func(self, edges):
        R1, R2 = edges.src['o'], edges.dst['o']
        x = torch.sum(R1 * R2, dim=-1)
        y = torch.cross(R1, R2)
        y = torch.norm(y, dim=-1)
        angle = torch.atan2(y, x)
        
        cbf = [f(angle) for f in self.sph_funcs]
        cbf = torch.stack(cbf, dim=1)  # [60, 7]
        cbf = cbf.repeat_interleave(self.num_radial, dim=1)  # [60, 42]
        sbf = edges.src['rbf_env'] * cbf  # [60, 42]
        sbf = self.dense_sbf(sbf)

        # [60, 8] * [60, 128] * [128, 8, 128] -> [60, 128]
        x_kj = torch.einsum("wj,wl,ijl->wi", sbf, edges.src['m'], self.W_bilin)
        # sbf [60, 42]
        return {'x_kj': x_kj}

    def reduce_func(self, nodes):
        # [20, 3, 128] -> [20, 128]
        return {'m_update': nodes.mailbox['x_kj'].sum(1)}

    def forward(self, g):
        g.apply_edges(self.edge_transfer)
        
        # node means edge in original graph
        # edge means node in original graph
        # node: d, rbf, o
        # edge: R, Z, h
        l_g = dgl.line_graph(g, backtracking=False, shared=True)
        l_g.update_all(self.msg_func, self.reduce_func)

        for k, v in l_g.ndata.items():
            g.edata[k] = v
        
        for k, v in l_g.edata.items():
            g.ndata[k] = v

        print(g)
        # print(type(g.ndata), type(g.edata))
        # print(type(l_g.ndata), type(l_g.edata))
        # print(l_g.ndata.keys())
        # msg_func: f(e_angle, u_w_e, W_bilinear), reduce_func: sum
        return g