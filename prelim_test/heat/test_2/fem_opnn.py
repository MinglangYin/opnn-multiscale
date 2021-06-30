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
# from dolfin import *

def get_boundary_index(V):
    dof_x = V.tabulate_dof_coordinates().reshape((V.dim(), -1))
    x = dof_x[:, 0]
    y = dof_x[:, 1]
    list_top = [x > 0.25-DOLFIN_EPS, x < 0.75+DOLFIN_EPS, abs(0.75-y)<DOLFIN_EPS]
    list_bot = [x > 0.25-DOLFIN_EPS, x < 0.75+DOLFIN_EPS, abs(0.25-y)<DOLFIN_EPS]
    list_left = [y > 0.25-DOLFIN_EPS, y < 0.75+DOLFIN_EPS, abs(0.25-x)<DOLFIN_EPS]
    list_right = [y > 0.25-DOLFIN_EPS, y < 0.75+DOLFIN_EPS, abs(0.75-x)<DOLFIN_EPS]
    list_inner = [y > 0.25+DOLFIN_EPS, y < 0.75-DOLFIN_EPS, x > 0.25+DOLFIN_EPS, x < 0.75-DOLFIN_EPS]

    ## top    
    indices_top = np.where( np.logical_and.reduce( list_top ))[0]
    indices_bot = np.where( np.logical_and.reduce( list_bot ))[0]
    indices_left = np.where( np.logical_and.reduce( list_left ))[0]
    indices_right = np.where( np.logical_and.reduce( list_right ))[0]
    indices_inner = np.where( np.logical_and.reduce( list_inner ))[0]

    return indices_top, indices_bot, indices_left, indices_right, indices_inner

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
    ########## DeepONet ##########
    model_path = './CheckPts/model_chkpts.pt'
    device = 'cuda'

    ## model initialization
    dim_br = [68, 100, 100, 100]
    dim_tr = [2, 100, 100, 100]
    model = opnn(dim_br, dim_tr).to(device)

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    ######### Fenics ##########
    # create mesh and define function space
    T = 2.0
    num_steps = 10
    dt = T/num_steps
    alpha = 3
    beta = 1.2

    mesh_size = 32
    mesh = UnitSquareMesh(mesh_size, mesh_size)
    V = FunctionSpace(mesh, 'P', 1)

    ## get indices
    indices_top, indices_bot, indices_left, indices_right, indices_inner = get_boundary_index(V) 
    dof_x = V.tabulate_dof_coordinates().reshape((V.dim(), -1))
    
    

    x_inner = dof_x[indices_inner]
    x_t = torch.tensor(x_inner, dtype=torch.float).to(device)
    
    x_size = x_inner.shape[0]

    ##
    u_D = Expression('1+x[0]*x[0]+alpha*x[1]*x[1]+beta*t', degree=2, alpha=alpha, beta=beta, t=0)

    def boundary(x, on_boundary):
        return on_boundary

    bc = DirichletBC(V, u_D, boundary)

    ## spaces
    u_n = project(u_D, V)
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Constant(beta - 2 - 2*alpha)

    F = u*v*dx + dt*dot(grad(u), grad(v))*dx - (u_n + dt*f)*v*dx
    a, L = lhs(F), rhs(F)

    ## Time stepping
    u = Function(V)
    t = 0

    x = V.tabulate_dof_coordinates()
    x = x.reshape((-1, mesh.geometry().dim()))
    t_vec = np.ones((x.shape[0], 1))*t
    u_tmp = np.array(u_n.vector())[:, None]
    txyu = []
    txyu.append(np.concatenate((t_vec, x, u_tmp), axis=1))

    for n in range(num_steps):
        ########## FEM ##########
        t += dt
        u_D.t = t

        solve(a==L, u, bc)
        u_n.assign(u)

        x_top = dof_x[indices_top]
        u_top = u.vector()[indices_top][:, None]
        xu_top = np.concatenate((x_top, u_top), axis=1)
        xu_top = xu_top[np.argsort(xu_top[:, 0])]
        x_top = xu_top[:, 0:2]
        u_top = xu_top[:, 2]

        x_bot = dof_x[indices_bot]
        u_bot = u.vector()[indices_bot][:, None]
        xu_bot = np.concatenate((x_bot, u_bot), axis=1)
        xu_bot = xu_bot[np.argsort(xu_bot[:, 0])]
        x_bot = xu_bot[:, 0:2]
        u_bot = xu_bot[:, 2]

        x_left = dof_x[indices_left]
        u_left = u.vector()[indices_left][:, None]
        xu_left = np.concatenate((x_left, u_left), axis=1)
        # print(xu_left)
        xu_left = xu_left[np.argsort(xu_left[:, 1])]
        # print(xu_left)
        x_left = xu_left[:, 0:2]
        u_left = xu_left[:, 2]

        x_right = dof_x[indices_right]
        u_right = u.vector()[indices_right][:, None]
        xu_right = np.concatenate((x_right, u_right), axis=1)
        xu_right = xu_right[np.argsort(xu_right[:, 1])]
        x_right = xu_right[:, 0:2]
        u_right = xu_right[:, 2]

        branch_input = np.concatenate((u_top, u_bot, u_left, u_right), axis=0)
        branch_input = np.repeat(branch_input[None, :], x_size, axis=0)

        # print(xu_top)
        # print(xu_bot)
        # print(xu_left)
        # print(xu_right)

        u_true = u.vector()[indices_inner]

        ######### DeepONet ##########
        x_b = torch.tensor(branch_input, dtype=torch.float).to(device)
        u_pred = model.forward(x_b, x_t)
        u_pred = to_numpy(u_pred)[:, 0]

        # print(u_pred)
        # print(u_true)
        print( ((u_true - u_pred)**2).mean() )



if __name__ == "__main__":
    main()