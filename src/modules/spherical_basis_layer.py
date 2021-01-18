import torch.nn as nn
import sympy as sym
import torch

from modules.basis_utils import bessel_basis, real_sph_harm
from modules.envelope import Envelope

class SphericalBasisLayer(nn.Module):
    def __init__(self,
                 num_spherical,
                 num_radial,
                 cutoff,
                 envelope_exponent=5):
        super(SphericalBasisLayer, self).__init__()

        assert num_radial <= 64
        self.num_radial = num_radial
        self.num_spherical = num_spherical
        self.cutoff = cutoff
        self.envelope = Envelope(envelope_exponent)

        # retrieve formulas
        self.bessel_formulas = bessel_basis(num_spherical, num_radial)  # x, [num_spherical, num_radial] sympy functions
        self.sph_harm_formulas = real_sph_harm(num_spherical)  # theta, [num_spherical, ] sympy functions
        self.sph_funcs = []
        self.bessel_funcs = []

        # convert to tensorflow functions
        x = sym.symbols('x')
        theta = sym.symbols('theta')
        # https://www.programcreek.com/python/example/38815/sympy.lambdify
        modules = {'sin': torch.sin, 'cos': torch.cos}
        for i in range(num_spherical):
            if i == 0:
                first_sph = sym.lambdify([theta], self.sph_harm_formulas[i][0], modules)(0)
                self.sph_funcs.append(lambda tensor: torch.zeros_like(tensor) + first_sph)
            else:
                self.sph_funcs.append(sym.lambdify([theta], self.sph_harm_formulas[i][0], modules))
            for j in range(num_radial):
                self.bessel_funcs.append(sym.lambdify([x], self.bessel_formulas[i][j], modules))
    
    # def forward(self, inputs):
    #     d, Angles, id_expand_kj = inputs  # 20, 60, 60
    #     d_scaled = d / self.cutoff
    #     rbf = [f(d_scaled) for f in self.bessel_funcs]
    #     rbf = torch.stack(rbf, dim=1)  # [20, 42]

    #     d_cutoff = self.envelope(d_scaled)  # [20,]
    #     rbf_env = d_cutoff[:, None] * rbf  # [20, 42]
    #     rbf_env = rbf_env[id_expand_kj.long()]  # [60, 42]

    #     cbf = [f(Angles) for f in self.sph_funcs]
    #     cbf = torch.stack(cbf, dim=1)  # [60, 7]
    #     cbf = cbf.repeat_interleave(self.num_radial, dim=1)  # [60, 42]

    #     return rbf_env * cbf

    def get_bessel_funcs(self):
        return self.bessel_funcs

    def get_sph_funcs(self):
        return self.sph_funcs