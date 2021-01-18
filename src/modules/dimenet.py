import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

from torch_scatter import scatter_add
from modules.activations import swish
from modules.bessel_basis_layer import BesselBasisLayer
from modules.spherical_basis_layer import SphericalBasisLayer
from modules.embedding_block import EmbeddingBlock
from modules.output_block import OutputBlock
from modules.interaction_block import InteractionBlock
from modules.basis_utils import calculate_interatomic_distances, calculate_neighbor_angles

class DimeNet(nn.Module):
    """
    DimeNet model.

    Parameters
    ----------
    emb_size
        Embedding size used throughout the model
    num_blocks
        Number of building blocks to be stacked
    num_bilinear
        Third dimension of the bilinear layer tensor
    num_spherical
        Number of spherical harmonics
    num_radial
        Number of radial basis functions
    envelope_exponent
        Shape of the smooth cutoff
    cutoff
        Cutoff distance for interatomic interactions
    num_before_skip
        Number of residual layers in interaction block before skip connection
    num_after_skip
        Number of residual layers in interaction block after skip connection
    num_dense_output
        Number of dense layers for the output blocks
    num_targets
        Number of targets to predict
    activation
        Activation function
    """
    def __init__(self,
                 emb_size,
                 num_blocks,
                 num_bilinear,
                 num_spherical,
                 num_radial,
                 cutoff=5.0,
                 envelope_exponent=5,
                 num_before_skip=1,
                 num_after_skip=2,
                 num_dense_output=3,
                 num_targets=12,
                 activation=swish):
        super(DimeNet, self).__init__()

        self.num_blocks = num_blocks

        # cosine basis function expansion layer
        self.rbf_layer = BesselBasisLayer(num_radial=num_radial,
                                          cutoff=cutoff,
                                          envelope_exponent=envelope_exponent)

        self.sbf_layer = SphericalBasisLayer(num_spherical=num_spherical,
                                             num_radial=num_radial,
                                             cutoff=cutoff,
                                             envelope_exponent=envelope_exponent)
        
        # embedding block
        self.emb_block = EmbeddingBlock(emb_size=emb_size,
                                        num_radial=num_radial,
                                        activation=activation)
        
        # output block
        self.output_blocks = nn.ModuleList({
            OutputBlock(emb_size=emb_size,
                        num_radial=num_radial,
                        num_dense=num_dense_output,
                        num_targets=num_targets,
                        activation=activation) for _ in range(num_blocks + 1)
        })

        # interaction block
        self.interaction_blocks = nn.ModuleList({
            InteractionBlock(emb_size=emb_size,
                             num_radial=num_radial,
                             num_spherical=num_spherical,
                             cutoff=cutoff,
                             envelope_exponent=envelope_exponent,
                             num_bilinear=num_bilinear,
                             num_before_skip=num_before_skip,
                             num_after_skip=num_after_skip,
                             sph_funcs=self.sbf_layer.get_sph_funcs(),
                             activation=activation) for _ in range(num_blocks)
        })
    
    def forward(self, g, data):
        Z, R = g.ndata['Z'], g.ndata['R']
        # add rbf features for each edge in one batch graph, [num_radial,]
        g = self.rbf_layer(g)
        # Embedding block
        g = self.emb_block(g)
        # Output block
        P = self.output_blocks[0](g)  # [batch_size, num_targets]
        # Interaction blocks
        for i in range(self.num_blocks):
            g = self.interaction_blocks[i](g)
            # P += self.output_blocks[i + 1]([x, rbf, dst])
            break
        
        # idnb_i = data['idnb_i']
        # idnb_j = data['idnb_j']
        # id3dnb_i = data['id3dnb_i']
        # id3dnb_j = data['id3dnb_j']
        # id3dnb_k = data['id3dnb_k']
        # id_expand_kj = data['id_expand_kj']
        # id_reduce_ji = data['id_reduce_ji']
        # batch_seg = data['batch_seg']

        # Calculate distances
        # Dij = calculate_interatomic_distances(R, idnb_i, idnb_j)  # [batch edges]

        # print('id3dnb_i')
        # print(type(id3dnb_i), id3dnb_i.size())
        # print(id3dnb_i)
        # print('id3dnb_j')
        # print(type(id3dnb_j), id3dnb_j.size())
        # print(id3dnb_j)
        # print('id3dnb_k')
        # print(type(id3dnb_k), id3dnb_k.size())
        # print(id3dnb_k)
        # print('id_expand_kj')
        # print(type(id_expand_kj), id_expand_kj.size())
        # print(id_expand_kj)
        # print('Dij')
        # print(type(Dij), Dij.size())
        # print(Dij)

        # Calculate angles
        # A_ijk = calculate_neighbor_angles(R, id3dnb_i, id3dnb_j, id3dnb_k)  # [batch edges,]
        # print('A_ijk')
        # print(type(A_ijk), A_ijk.size())
        # print(A_ijk)
        # sbf = self.sbf_layer((Dij, A_ijk, id_expand_kj))  # [batch edges, num_radial * num_spherical]

        # print('sbf')
        # print(type(sbf), sbf.size())
        # print(sbf)
        return 0

        # Interaction blocks
        for i in range(self.num_blocks):
            x = self.int_blocks[i]((x, rbf, sbf, id_expand_kj, id_reduce_ji))
            P += self.output_blocks[i + 1]([x, rbf, dst])

        P = scatter_add(P, batch_seg, dim=0)

        return P