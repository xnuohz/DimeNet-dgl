import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.envelope import Envelope

class BesselBasisLayer(nn.Module):
    def __init__(self,
                 num_radial,
                 cutoff,
                 envelope_exponent=5):
        super(BesselBasisLayer, self).__init__()

        self.cutoff = cutoff
        self.num_radial = num_radial
        self.envelope = Envelope(envelope_exponent)
        self.frequencies = nn.Parameter(np.pi * torch.arange(1, num_radial + 1))

    def forward(self, g):
        R = g.ndata['R']
        g.edata['rbf'] = torch.empty(g.num_edges(), self.num_radial)
        src, dst = g.edges(order='eid')
        for i in range(g.num_edges()):
            R_src, R_dst = R[src[i]], R[dst[i]]
            dist = torch.sqrt(F.relu(torch.sum((R_src - R_dst) ** 2)))
            d_scaled = dist / self.cutoff
            # Necessary for proper broadcasting behaviour
            d_scaled = torch.unsqueeze(d_scaled, -1)
            d_cutoff = self.envelope(d_scaled)
            g.edata['rbf'][g.edge_ids(src[i].item(), dst[i].item())] = d_cutoff * torch.sin(self.frequencies * d_scaled)
        return g