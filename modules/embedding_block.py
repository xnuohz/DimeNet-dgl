import torch
import torch.nn as nn

class EmbeddingBlock(nn.Module):
    def __init__(self,
                emb_size,
                num_radial,
                activation=None):
        super(EmbeddingBlock, self).__init__()

        self.activation = activation
        self.embedding = nn.Embedding(100, emb_size, padding_idx=0)
        self.dense_rbf = nn.Linear(num_radial, emb_size)
        self.dense = nn.Linear(emb_size * 3, emb_size)

    def edge_init(self, edges):
        rbf = self.dense_rbf(edges.data['rbf'])

        if self.activation is not None:
            rbf = self.activation(rbf)

        m = torch.cat([edges.src['h'], edges.dst['h'], rbf], dim=-1)
        m = self.dense(m)

        if self.activation is not None:
            m = self.activation(m)

        return {'m': m}

    def forward(self, g):
        g.apply_nodes(lambda nodes: {'h': self.embedding(nodes.data['Z'])})
        g.apply_edges(self.edge_init)
        return g