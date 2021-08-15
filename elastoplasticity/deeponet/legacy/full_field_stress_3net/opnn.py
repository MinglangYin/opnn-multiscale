"""
Author: Minglang Yin, minglang_yin@brown.edu
Multifidelity Architecture 
"""
import torch
import torch.nn as nn
#### FNN
class opnn(nn.Module):
    def __init__(self, branch_dim, trunk_dim, out_dim):
        super(opnn, self).__init__()
        self.branch_dim = branch_dim
        self.trunk_dim = trunk_dim
        self.z_dim = trunk_dim[-1]
        
        ## Net 1
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
        self._branch1 = nn.Sequential(*modules)

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
        self._trunk1 = nn.Sequential(*modules)
        self._out_layer1 = nn.Linear(self.z_dim, 1, bias = True)

        ## Net 2
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
        self._branch2 = nn.Sequential(*modules)

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
        self._trunk2 = nn.Sequential(*modules)
        self._out_layer2 = nn.Linear(self.z_dim, 2, bias = True)


    def forward(self, f, x):
        y_br1 = self._branch1(f)
        y_tr1 = self._trunk1(x) 
        y_out1 = self._out_layer1(y_br1*y_tr1)

        y_br2 = self._branch2(f)
        y_tr2 = self._trunk2(x) 
        y_out2 = self._out_layer2(y_br2*y_tr2)

        return torch.cat( (y_out1, y_out2), dim=1)