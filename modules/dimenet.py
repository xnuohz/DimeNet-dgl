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



def calculate_interatomic_distances(R, idx_i, idx_j):
    Ri, Rj = R[idx_i], R[idx_j]
    Dij = torch.sqrt(F.relu(torch.sum((Ri - Rj) ** 2), -1))
    return Dij


def calculate_neighbor_angles(R, id3_i, id3_j, id3_k):
    Ri, Rj, Rk = R[id3_i], R[id3_j], R[id3_k]
    R1, R2 = Rj - Ri, Rk - Rj
    x = torch.sum(R1 * R2, axis=-1)
    y = torch.cross(R1, R2)
    y = torch.norm(y, axis=-1)
    angle = torch.atan2(y, x)
    return angle


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

        # self.sbf_layer = SphericalBasisLayer(num_spherical=num_spherical,
        #                                      num_radial=num_radial,
        #                                      cutoff=cutoff,
        #                                      envelope_exponent=envelope_exponent)
        
        # embedding block
        # self.emb_block = EmbeddingBlock(emb_size=emb_size,
        #                                 num_radial=num_radial,
        #                                 activation=activation)
        
        # output block
        # self.output_blocks = nn.ModuleList({
        #     OutputBlock(emb_size=emb_size,
        #                 num_radial=num_radial,
        #                 num_dense=num_dense_output,
        #                 num_targets=num_targets,
        #                 activation=activation) for _ in range(num_blocks + 1)
        # })

        # interaction block
        # self.interaction_blocks = nn.ModuleList({
        #     InteractionBlock(emb_size=emb_size,
        #                      num_radial=num_radial,
        #                      num_spherical=num_spherical,
        #                      num_bilinear=num_bilinear,
        #                      num_before_skip=num_before_skip,
        #                      num_after_skip=num_after_skip,
        #                      activation=activation) for _ in range(num_blocks)
        # })
    
    def forward(self, g):
        Z, R = g.ndata['Z'], g.ndata['R']

        adj = g.adj(scipy_fmt='csr')
        degree = g.out_degrees()

        dst, src = g.edges()
        dst, src = dst.type(torch.long), src.type(torch.long)
        ntriplets = degree[src]
        
        id3ynb_i = dst.repeat(ntriplets)
        id3ynb_j = src.repeat(ntriplets)
        id3ynb_k = adj[src].nonzero()[1]

        print(id3ynb_k)
        
        # idnb_i = edgeid_to_target.type(torch.long)
        # idnb_j = edgeid_to_source.type(torch.long)

        idnb_i = edgeid_to_target
        idnb_j = edgeid_to_source

        # all i -> j -> k
        id3_y_to_d = (id3ynb_i != id3ynb_k).nonzero()
        id3dnb_i = id3ynb_i[id3_y_to_d]
        id3dnb_j = id3ynb_j[id3_y_to_d]
        id3dnb_k = id3ynb_k[id3_y_to_d]

        # id_expand_kj = adj[edgeid_to_source, :].data[id3_y_to_d]

        # id_reduce_ji = inputs.id_reduce_ji
        # batch_seg = inputs.batch_seg

        # Calculate distances
        Dij = calculate_interatomic_distances(R, idnb_i, idnb_j)  # float value
        rbf = self.rbf_layer(Dij)  # [num_radial,]

        # Calculate angles
        # A_ijk = calculate_neighbor_angles(R, id3dnb_i, id3dnb_j, id3dnb_k)
        # sbf = self.sbf_layer((Dij, A_ijk, id_expand_kj))
        return 0

        # Embedding block
        x = self.emb_block((Z, rbf, idnb_i, idnb_j))
        P = self.output_blocks[0]((x, rbf, idnb_i))

        # Interaction blocks
        for i in range(self.num_blocks):
            x = self.int_blocks[i]((x, rbf, sbf, id_expand_kj, id_reduce_ji))
            P += self.output_blocks[i + 1]([x, rbf, idnb_i])

        P = scatter_add(P, batch_seg, dim=0)

        return P