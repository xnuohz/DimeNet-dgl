import torch
import torch.nn as nn

class EmbeddingBlock(nn.Module):
    def __init__(self,
                emb_size,
                num_radial,
                activation=None):
        super(EmbeddingBlock, self).__init__()

        self.emb_size = emb_size
        self.activation = activation
        self.embedding = nn.Embedding(100, emb_size, padding_idx=0)
        self.dense_rbf = nn.Linear(num_radial, emb_size)
        self.dense = nn.Linear(emb_size * 3, emb_size)

    def forward(self, inputs):
        Z, rbf, idnb_i, idnb_j = inputs

        rbf = self.dense_rbf(rbf)
        if self.activation is not None:
            rbf = self.activation(rbf)
        x = self.embedding(Z)

        x1 = x[idnb_i]
        x2 = x[idnb_j]

        x = torch.cat((x1, x2, rbf), dim=-1)
        x = self.dense(x)
        if self.activation is not None:
            x = self.activation(x)

        return x