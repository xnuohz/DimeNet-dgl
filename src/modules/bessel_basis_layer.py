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

    def add_rbf_in_edge(self, edges):
        d_scaled = edges.data['d'] / self.cutoff
        # Necessary for proper broadcasting behaviour
        d_scaled = torch.unsqueeze(d_scaled, -1)
        d_cutoff = self.envelope(d_scaled)
        return {'rbf': d_cutoff * torch.sin(self.frequencies * d_scaled)}

    def forward(self, g):
        g.apply_edges(self.add_rbf_in_edge)
        return g