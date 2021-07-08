"""
Minglang Yin, minglang_yin@brown.edu
Multiscale Modeling: FEM + DeepONet
Schwarz alternating method
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from dolfin import *
from mshr import *
from scipy.interpolate import interp2d

def get_bound_coord(archor_p1, archor_p2, dim):
    x_min, y_min = archor_p1[0], archor_p1[1]
    x_max, y_max = archor_p2[0], archor_p2[1]

    vline = np.linspace(x_min, x_max, dim)[:, None]
    hline = np.linspace(y_min, y_max, dim)[:, None]
    ones = np.ones((dim, 1))

    top = np.concatenate( (vline, ones*y_max), axis=1 )
    bot = np.concatenate( (vline, ones*y_min), axis=1 )
    left = np.concatenate( (ones*x_min, hline), axis=1 )
    right = np.concatenate( (ones*x_max, hline), axis=1 )

    return top, bot, left, right

def data2exp(data, V):
    x, y, values = data[:,0], data[:,1], data[:,2]
    interpolant = interp2d(x, y, values, kind='cubic', copy=False, bounds_error=True)

    class ExpressionFromScipyFunction(UserExpression):
        def __init__(self, f, **kwargs):
            self._f = f
            UserExpression.__init__(self, **kwargs)
        def eval(self, values, x):
            values[:] = self._f(*x)

    u = ExpressionFromScipyFunction(interpolant, element=V.ufl_element())
    return u

def get_true_u(coord):
    return 1+coord[:, 0]**2+2*coord[:, 1]**2

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

def main():
    ## parameter
    niter = 20
    eps = 0.001
    gemma1_x, gemma1_y = 0.4, 0.6
    gemma2_x, gemma2_y = 0.3, 0.7
    model_path = './CheckPts/model_chkpts.pt'
    device = 'cuda'

    ## define the non-conforming mesh
    box = Rectangle(Point(0,0), Point(1.0, 1.0))
    box_in = Rectangle(Point(gemma1_x, gemma1_x), Point(gemma1_y, gemma1_y))
    omega_s = box - box_in

    omega_p = Rectangle(Point(gemma2_x, gemma2_x), Point(gemma2_y, gemma2_y))
    
    mesh_s = generate_mesh(omega_s, 16)
    mesh_p = generate_mesh(omega_p, 16)

    ## function space
    V_s = FunctionSpace(mesh_s, 'P', 1)
    V_p = FunctionSpace(mesh_p, 'P', 1)

    ## define boundary condition
    u_true = Expression('1 + x[0]*x[0] + 2*x[1]*x[1]', degree=2)
    u_o = Expression('0', degree=1)

    s_go = 'on_boundary && x[0]>=0-DOLFIN_EPS && x[0]<=1+DOLFIN_EPS &&\
            x[1]>=0-DOLFIN_EPS && x[1]<=1+DOLFIN_EPS' # mesh_s outer boundary, need to be consistent
    s_g1 = 'on_boundary && x[0]>=0.4-DOLFIN_EPS && x[0]<=0.6+DOLFIN_EPS &&\
            x[1]>=0.4-DOLFIN_EPS && x[1]<=0.6+DOLFIN_EPS' # mesh_s on Gemma 1
    p_g2 = 'on_boundary && x[0]>=0.3-DOLFIN_EPS && x[0]<=0.7+DOLFIN_EPS &&\
            x[1]>=0.3-DOLFIN_EPS && x[1]<=0.7+DOLFIN_EPS' # mesh_p on Gemma 2
    mesh_s_go = DirichletBC(V_s, u_true, s_go)
    mesh_s_g1 = DirichletBC(V_s, u_o, s_g1)

    ## true solution
    u_es = interpolate(u_true, V_s)
    u_ep = interpolate(u_true, V_p)
    
    ## boundary coordinates
    dim = 17
    top_o, bot_o, left_o, right_o = get_bound_coord([0, 0], [1.0, 1.0], dim)
    top_g1, bot_g1, left_g1, right_g1 = get_bound_coord([gemma1_x, gemma1_x], [gemma1_y, gemma1_y], dim)
    top_g2, bot_g2, left_g2, right_g2 = get_bound_coord([gemma2_x, gemma2_x], [gemma2_y, gemma2_y], dim)

    g1_coord = np.concatenate((top_g1, bot_g1, left_g1, right_g1), axis=0)
    g2_coord = np.concatenate((top_g2, bot_g2, left_g2, right_g2), axis=0)
    o_coord = np.concatenate((top_o, bot_o, left_o, right_o), axis=0)

    x_g2_tensor = torch.tensor(g2_coord, dtype=torch.float).to(device)
    x_g1_tensor = torch.tensor(g1_coord, dtype=torch.float).to(device)

    ## inner points of the mesh_p
    x_p_2d = mesh_p.coordinates()
    dim_p = x_p_2d.shape[0]
    x_p_tensor = torch.tensor(x_p_2d, dtype=torch.float).to(device)

    u_p_ini = np.zeros((x_p_2d.shape[0], dim*4))
    u_p_tensor = torch.tensor(u_p_ini, dtype=torch.float).to(device)

    u_p_g2_ini = np.zeros((dim*4, dim*4))
    u_p_g2_tensor = torch.tensor(u_p_g2_ini, dtype=torch.float).to(device)

    u_s_out = get_true_u(o_coord)[:, None]
    
    ## mesh_s, define the variational form
    u_s = TrialFunction(V_s)
    v_s = TestFunction(V_s)
    f_s = Constant(-6.0)
    a_s = dot(grad(u_s), grad(v_s))*dx
    L_s = f_s*v_s*dx
    u_s = Function(V_s)

    ## model initialization
    dim_br = [68, 75, 75, 75]
    dim_tr = [2, 75, 75, 75]
    model = opnn(dim_br, dim_tr).to(device)

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    ## Schwarz alternating method
    tau_n1 = 0.5
    u_s_g1_new = np.zeros((dim*4, 1))
    u_s_g2_old = np.zeros((dim*4, 1))
    u_s_g2_new = np.zeros((dim*4, 1))

    u_p_g1_old = np.zeros((dim*4, 1))
    u_p_g2_old = np.zeros((dim*4, 1))
    u_p_g2_new = np.zeros((dim*4, 1))

    err_his = [] ## [err_in, err_out]
    for iter in range(0, niter):
        if iter == 0:
            solve(a_s == L_s, u_s, [mesh_s_go, mesh_s_g1])
            u_p = to_numpy(model.forward(u_p_tensor, x_p_tensor))

            for id, coord in enumerate(g2_coord):
                u_s_g2_old[id] = u_s(coord)
            
            u_p_g2_old = to_numpy(model.forward(u_p_g2_tensor, x_g2_tensor))

        else:
            ## step a
            u_p_g2_stack = np.tile(u_p_g2_old.reshape(1, -1), [dim*4, 1])
            u_p_g2_tensor = torch.tensor(u_p_g2_stack, dtype=torch.float).to(device)

            u_s_g1_new = to_numpy(model.forward(u_p_g2_tensor, x_g1_tensor))

            x_s_intp = np.concatenate((g1_coord, o_coord), axis=0)
            u_s_intp = np.concatenate((u_s_g1_new, u_s_out), axis=0)
            u_s_bc_exp = data2exp(np.concatenate((x_s_intp, u_s_intp), axis=1), V_s)

            ## step b
            mesh_s_g1 = DirichletBC(V_s, u_s_bc_exp, s_g1)
            solve(a_s == L_s, u_s, [mesh_s_go, mesh_s_g1])

            ## step c, d
            for id, coord in enumerate(g2_coord):
                u_s_g2_new[id] = u_s(coord)

            ## step e (relaxation)
            if iter == 1:
                Q_n1 = np.copy(u_s_g2_new) #np.zeros((dim*4, 1))
            else:
                Q_n1 = np.copy(Q_n)
            Q_n = u_s_g2_new - u_p_g2_old
            
            tau_n = tau_n1 + (tau_n1 - 1)*((Q_n1 - Q_n)*Q_n).sum()/(((Q_n1 - Q_n)**2).sum())
            u_p_g2_new = tau_n*u_p_g2_old + (1 - tau_n)*u_s_g2_new

            ## step f
            u_p_stack = np.tile(u_p_g2_new.reshape(1, -1), [dim_p, 1])
            u_p_tensor = torch.tensor(u_p_stack, dtype=torch.float).to(device)
            u_p = to_numpy(model.forward(u_p_tensor, x_p_tensor))

            ## update
            u_p_g2_old = np.copy(u_p_g2_new)
            tau_n1 = np.copy(tau_n)

        ## error 
        u_p_intp_exp = data2exp(np.concatenate((x_p_2d, u_p), axis=1), V_p)
        u_p_intp = interpolate(u_p_intp_exp, V_p)

        error_s = ((np.array(u_es.vector()) - np.array(u_s.vector()))**2).mean()
        error_p = ((np.array(u_ep.vector()) - np.array(u_p_intp.vector()))**2).mean()
        err_his.append( np.array([error_s, error_p]) )
        print(f'iter: {iter}, error_s: {error_s:3g}, error_p: {error_p:3g}')

        # if error_p <= eps and error_s <= eps:
        #     break

    err_his = np.array(err_his)

    ## figures u_s
    fig = plt.figure(constrained_layout=False, figsize=(11, 3))
    gs = fig.add_gridspec(1, 3)
    ax = fig.add_subplot(gs[0])
    h = plot(u_s)
    plt.colorbar(h)
    ax.set_title('prediction')

    ax = fig.add_subplot(gs[1])
    h = plot(u_es)
    plt.colorbar(h)
    ax.set_title('true')

    ax = fig.add_subplot(gs[2])
    h = plot(abs(u_s - u_es))
    plt.colorbar(h)
    ax.set_title('err')

    plt.savefig("u_s.png")
    plt.close()

    ## figures u_p
    fig = plt.figure(constrained_layout=False, figsize=(11, 3))
    gs = fig.add_gridspec(1, 3)
    ax = fig.add_subplot(gs[0])
    h = plot(u_p_intp)
    plt.colorbar(h)
    ax.set_title('prediction')

    ax = fig.add_subplot(gs[1])
    h = plot(u_ep)
    plt.colorbar(h)
    ax.set_title('true')

    ax = fig.add_subplot(gs[2])
    h = plot(abs(u_p_intp - u_ep))
    plt.colorbar(h)
    ax.set_title('err')

    plt.savefig("u_p.png")
    plt.close()

    ## mesh 
    fig = plt.figure(constrained_layout=False, figsize=(11, 3))
    gs = fig.add_gridspec(1, 3)
    ax = fig.add_subplot(gs[0])
    plot(mesh_s, alpha=0.5)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    ax = fig.add_subplot(gs[1])
    plot(mesh_p, alpha=0.5)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    ax = fig.add_subplot(gs[2])
    plot(mesh_p, alpha=0.5)
    plot(mesh_s, alpha=0.5)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.savefig("mesh.png")
    plt.close()

    ## figure convergence
    fig = plt.figure(constrained_layout=False, figsize=(4, 4))
    gs = fig.add_gridspec(1, 1)
    ax = fig.add_subplot(gs[0])
    ax.plot(err_his[:, 0], color='blue', label='$u_{s}$ (FEM)')
    ax.plot(err_his[:, 1], color='red', label='$u_{p}$ (DeepONet)')
    ax.axhline(y=eps, linestyle='--', color='black',  label='$\epsilon$')

    ax.set_xlabel('# iter')
    ax.set_ylabel('$L_{2}$ error')
    ax.set_yscale('log')
    ax.legend()
    plt.savefig("convergence.png")
    plt.close()

if __name__ == "__main__":
    main()