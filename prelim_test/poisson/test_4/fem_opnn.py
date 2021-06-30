"""
Author: Minglang Yin. minglang_yin@brown.edu

Coupling a FEM solver with a DeepONet for poisson equation

div(grad(u)) = -f, Omage = [0, 1]*[0, 1], f = 1
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from fenics import *
from dolfin import *

def to_numpy(input):
    if isinstance(input, torch.Tensor):
        return input.detach().cpu().numpy()
    elif isinstance(input, np.ndarray):
        return input
    else:
        raise TypeError('Unknown type of input, expected torch.Tensor or '\
            'np.ndarray, but got {}'.format(type(input)))

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
        y_out = self._out_layer(y_br*y_tr)
        return y_out

def main():
    ## parameters
    model_path = './CheckPts/model_chkpts.pt'
    device = 'cuda'

    ## create mesh and define function space
    num_cases = 100
    mesh_size = 32
    mesh = UnitSquareMesh(mesh_size, mesh_size)
    V = FunctionSpace(mesh, 'P', 1)

    ## define the variational form
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Constant(-6.0)
    a = dot(grad(u), grad(v))*dx
    L = f*v*dx

    ## boundary condition
    u_D = Expression('a + b*x[0]*x[0] + c*x[1]*x[1]', a=1, b=1, c=2, degree=2)

    def boundary(x, on_boundary):
        return on_boundary
    
    bc = DirichletBC(V, u_D, boundary)

    ########## FEM solver ##########
    u = Function(V)
    solve(a == L, u, bc)

    ## plot 
    plot(u)
    name = 'u.png'
    plt.savefig(name)
    plt.close()

    ########## DeepONet ##########

    ## One way to extract data 
    dof_x =  V.tabulate_dof_coordinates().reshape((V.dim(), -1))
    x = dof_x[:, 0]
    y = dof_x[:, 1]
    list_top = [x > 0.25-DOLFIN_EPS, x < 0.75+DOLFIN_EPS, abs(0.75-y)<DOLFIN_EPS]
    list_bot = [x > 0.25-DOLFIN_EPS, x < 0.75+DOLFIN_EPS, abs(0.25-y)<DOLFIN_EPS]
    list_left = [y > 0.25-DOLFIN_EPS, y < 0.75+DOLFIN_EPS, abs(0.25-x)<DOLFIN_EPS]
    list_right = [y > 0.25-DOLFIN_EPS, y < 0.75+DOLFIN_EPS, abs(0.75-x)<DOLFIN_EPS]
    list_inner = [y > 0.25+DOLFIN_EPS, y < 0.75-DOLFIN_EPS, x > 0.25+DOLFIN_EPS, x < 0.75-DOLFIN_EPS]

    ## top    
    indices_top = np.where( np.logical_and.reduce( list_top ))[0]
    x_top = dof_x[indices_top]
    u_top = u.vector()[indices_top][:, None]
    xu_top = np.concatenate((x_top, u_top), axis=1)
    xu_top = xu_top[np.argsort(xu_top[:, 0])]
    x_top = xu_top[:, 0:2]
    u_top = xu_top[:, 2]

    ## bottom
    indices_bot = np.where( np.logical_and.reduce( list_bot ))[0]
    x_bot = dof_x[indices_bot]
    u_bot = u.vector()[indices_bot][:, None]
    xu_bot = np.concatenate((x_bot, u_bot), axis=1)
    xu_bot = xu_bot[np.argsort(xu_bot[:, 0])]
    x_bot = xu_bot[:, 0:2]
    u_bot = xu_bot[:, 2]

    ## left side
    indices_left = np.where( np.logical_and.reduce( list_left ))[0]
    x_left = dof_x[indices_left]
    u_left = u.vector()[indices_left][:, None]
    xu_left = np.concatenate((x_left, u_left), axis=1)
    xu_left = xu_left[np.argsort(xu_left[:, 1])]
    x_left = xu_left[:, 0:2]
    u_left = xu_left[:, 2]

    ## right side
    indices_right = np.where( np.logical_and.reduce( list_right ))[0]
    x_right = dof_x[indices_right]
    u_right = u.vector()[indices_right][:, None]
    xu_right = np.concatenate((x_right, u_right), axis=1)
    xu_right = xu_right[np.argsort(xu_right[:, 1])]
    x_right = xu_right[:, 0:2]
    u_right = xu_right[:, 2]

    ## inner
    indices_inner = np.where( np.logical_and.reduce( list_inner ))[0]
    x_inner = dof_x[indices_inner]
    u_true = u.vector()[indices_inner]

    branch_input = np.concatenate((u_top, u_bot, u_left, u_right), axis=0)

    print(x_top, x_bot, x_left, x_right)

    x_size = x_inner.shape[0]
    branch_input = np.repeat(branch_input[None, :], x_size, axis=0)

    ## model initialization
    dim_br = [68, 100, 100, 100]
    dim_tr = [2, 100, 100, 100]
    model = opnn(dim_br, dim_tr).to(device)

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    x_b = torch.tensor(branch_input, dtype=torch.float).to(device)
    x_t = torch.tensor(x_inner, dtype=torch.float).to(device)

    u_pred = model.forward(x_b, x_t)
    u_pred = to_numpy(u_pred)[:, 0]

    print(u_pred)
    print(u_true)
    print( ((u_true - u_pred)**2).mean() )


if __name__ == "__main__":
    main()