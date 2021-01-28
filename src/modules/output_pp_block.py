import torch.nn as nn
import dgl
import dgl.function as fn

class OutputPPBlock(nn.Module):
    def __init__(self,
                 emb_size,
                 out_emb_size,
                 num_radial,
                 num_dense,
                 num_targets,
                 activation=None,
                 extensive=True):
        super(OutputPPBlock, self).__init__()

        self.activation = activation
        self.extensive = extensive
        self.dense_rbf = nn.Linear(num_radial, emb_size, bias=False)
        self.up_projection = nn.Linear(emb_size, out_emb_size, bias=False)
        self.dense_layers = nn.ModuleList([
            nn.Linear(out_emb_size, out_emb_size) for _ in range(num_dense)
        ])
        self.dense_final = nn.Linear(out_emb_size, num_targets, bias=False)
        self.reset_params()
    
    def reset_params(self):
        nn.init.xavier_normal_(self.dense_rbf.weight)
        nn.init.xavier_normal_(self.up_projection.weight)
        nn.init.zeros_(self.dense_final.weight)

    @profile
    def forward(self, g):
        with g.local_scope():
            g.edata['tmp'] = g.edata['m'] * self.dense_rbf(g.edata['rbf'])
            g.update_all(fn.copy_e('tmp', 'x'), fn.sum('x', 't'))
            g.ndata['t'] = self.up_projection(g.ndata['t'])
            # g.apply_nodes(self.node_udf)
            for layer in self.dense_layers:
                g.ndata['t'] = layer(g.ndata['t'])
                if self.activation is not None:
                    g.ndata['t'] = self.activation(g.ndata['t'])
            g.ndata['t'] = self.dense_final(g.ndata['t'])
            return dgl.readout_nodes(g, 't', op='sum' if self.extensive else 'mean')