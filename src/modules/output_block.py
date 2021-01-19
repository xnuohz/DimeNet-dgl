import torch.nn as nn
import dgl
import dgl.function as fn

class OutputBlock(nn.Module):
    def __init__(self,
                 emb_size,
                 num_radial,
                 num_dense,
                 num_targets,
                 activation=None):
        super(OutputBlock, self).__init__()

        self.activation = activation
        self.dense_rbf = nn.Linear(num_radial, emb_size, bias=False)
        self.dense_layers = nn.ModuleList([
            nn.Linear(emb_size, emb_size) for _ in range(num_dense)
        ])
        self.dense_final = nn.Linear(emb_size, num_targets, bias=False)
    
    def node_udf(self, nodes):
        t = nodes.data['t']
        for layer in self.dense_layers:
            t = layer(t)
            if self.activation is not None:
                t = self.activation(t)
        
        t = self.dense_final(t)
        return {'p': t}

    def forward(self, g):
        with g.local_scope():
            g.edata['tmp'] = g.edata['m'] * self.dense_rbf(g.edata['rbf'])
            g.update_all(fn.copy_e('tmp', 'x'), fn.sum('x', 't'))
            g.apply_nodes(self.node_udf)
            return dgl.readout_nodes(g, 'p')