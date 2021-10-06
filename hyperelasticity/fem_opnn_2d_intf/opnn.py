"""
Author: Minglang Yin, minglang_yin@brown.edu
"""
import torch
import torch.nn as nn
#### FNN
class opnn(nn.Module):
    def __init__(self, branch_dim, trunk_dim):
        super(opnn, self).__init__()
        self.branch_dim = branch_dim
        self.trunk_dim = trunk_dim
        self.z_dim = trunk_dim[-1]
         ## build branch net
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

         ## build trunk net
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
        y_out = torch.sum(y_tr*y_br, dim=1)[:, None]
        return y_out
