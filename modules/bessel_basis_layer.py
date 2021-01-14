import numpy as np
import torch
import torch.nn as nn

from modules.envelope import Envelope

class BesselBasisLayer(nn.Module):
    def __init__(self,
                 num_radial,
                 cutoff,
                 envelope_exponent=5):
        super(BesselBasisLayer, self).__init__()

        self.cutoff = cutoff
        self.envelope = Envelope(envelope_exponent)
        self.frequencies = nn.Parameter(np.pi * torch.arange(1, num_radial + 1))
    
    def forward(self, inputs):
        d_scaled = inputs / self.cutoff
        # Necessary for proper broadcasting behaviour
        d_scaled = torch.unsqueeze(d_scaled, -1)
        d_cutoff = self.envelope(d_scaled)
        return d_cutoff * torch.sin(self.frequencies * d_scaled)