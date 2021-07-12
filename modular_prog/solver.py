import torch
import torch.nn as nn

from fenics import *

class solver_fem(object):
    def __init__(self, a, L, u):
        self.a = a
        self.L = L
        self.u = u

    def forward(self, bc):
        solve(self.a == self.L, self.u, bc)
        return self.u

    def interpolate(self, x):
        u_intp = []
        for x_tmp in x:
            u_intp.append(u(x_tmp))
        return np.array(u_intp)

class opnn(nn.Module):
    def __init__(self, branch_dim, trunk_dim):
        super(opnn, self).__init__()
        self.branch_dim = branch_dim
        self.trunk_dim = trunk_dim
        self.z_dim = trunk_dim[-1]
        
        modules = []
        for i, h_dim in enumerate(branch_dim):
            if i == 0:
                in_channels = h_dim
            else:
                modules.append(nn.Sequential(
                    nn.Linear(in_channels, h_dim),
                    nn.Tanh()
                    )
                )
                in_channels = h_dim
        self._branch = nn.Sequential(*modules)

        modules = []
        for i, h_dim in enumerate(trunk_dim):
            if i == 0:
                in_channels = h_dim
            else:
                modules.append(nn.Sequential(
                    nn.Linear(in_channels, h_dim),
                    nn.Tanh()
                    )
                )
                in_channels = h_dim
        self._trunk = nn.Sequential(*modules)
        self._out_layer = nn.Linear(self.z_dim, 1, bias = True)

    def forward(self, f, x):
        y_br = self._branch(f)
        y_tr = self._trunk(x) 
        y_out = self._out_layer(y_br*y_tr)
        return y_out

def get_true_u(coord):
    return 1+coord[:, 0]**2+2*coord[:, 1]**2


