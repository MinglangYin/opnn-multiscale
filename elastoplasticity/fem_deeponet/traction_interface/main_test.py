"""
Minglang Yin, minglang_yin@brown.edu
Coupling DeepONet with Linear Finite Element Model
Note: 
    s11_model takes scaled displacement as input (disp = disp_old * 100)
    The output stress is scaled by 10 (disp = disp_old * 100)
"""
import torch
import numpy as np
import matplotlib.pyplot as plt

from fenics import *
from ufl import nabla_div
from mshr import *
from scipy.interpolate import interp2d
from scipy.interpolate import Rbf
from opnn import *
from utils import *
from mpl_toolkits.axes_grid1 import make_axes_locatable

def epsilon(u):
    return 0.5*(nabla_grad(u)+nabla_grad(u).T)

def sigma(u):
    return lambda_*nabla_div(u)*Identity(d) + 2*mu*epsilon(u)

def normal_unit(x):
    norm = np.linalg.norm(x, ord=2, axis=1)[:, None]
    return x/norm

def find_outside(x):
    list = []
    for pt in x:
        if (pt[0]**2+pt[1]**2) > 0.1:
            list.append(pt)
    return np.array(list)

## parameters
mu = 1/(2*1.3)   # E/(2*(1+nu))
lambda_ = 0.3/(1.3*0.4)   # E*nu/((1+nu)*(1-2*nu))

eps = 1e-6
niter = 20
theta = 0.5
r0 = 0.3

U_scaling = 100
T_scaling = 100

########## DeepONet ##########
device = 'cuda'
dim_br = [80, 100, 100, 100]
dim_tr = [2, 40, 40, 100]

## initialization
model_tx = opnn(dim_br, dim_tr).to(device)
model_tx = model_tx.float()
tx_path = "./tx_model/model_chkpts.pt"
checkpoint = torch.load(tx_path)
model_tx.load_state_dict(checkpoint["model_state_dict"])

model_ty = opnn(dim_br, dim_tr).to(device)
model_ty = model_ty.float()
ty_path = "./ty_model/model_chkpts.pt"
checkpoint = torch.load(ty_path)
model_ty.load_state_dict(checkpoint["model_state_dict"])

## interface and lines
x_intf = np.loadtxt("x_intf_ext.dat")
n_intf = normal_unit(x_intf) ## assume circle
x_t_tensor = torch.tensor(x_intf, dtype=torch.float).to(device) ## tensorize the input of the trunk net
dim_intf = x_intf.shape[0]

########## FEM ##########
## mesh
square = Rectangle(Point(-1.0, -1.0), Point(1.0, 1.0))
circle = Circle(Point(0.0, 0.0), 0.3)
mesh = generate_mesh(square-circle, 32)
mesh_coord = mesh.coordinates()
x_out = find_outside(mesh_coord)
n_out = normal_unit(x_out)

## Space and Boundary
W = TensorFunctionSpace(mesh, "P", 2)
V = VectorFunctionSpace(mesh, "P", 2)

boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundaries.set_all(0)
top = AutoSubDomain(lambda x: x[1] > 1.0-1e-5)
top.mark(boundaries, 1)
mesh_intf = AutoSubDomain(lambda x: sqrt(x[0]*x[0]+x[1]*x[1]) < 0.3+1e-5)
mesh_intf.mark(boundaries, 2)
ds_fem = ds(subdomain_data=boundaries)

def clamped_boundary(x, on_boundary):
    return on_boundary and x[1] < -1 + eps
bc_bot = DirichletBC(V, Constant((0, 0)), clamped_boundary)

# datafiles = ["datafile/AbaqusOutput_Job-Example-10.npy",
#             "datafile/AbaqusOutput_Job-Example-20.npy",
#             "datafile/AbaqusOutput_Job-Example-30.npy",
#             "datafile/AbaqusOutput_Job-Example-40.npy",
#             "datafile/AbaqusOutput_Job-Example-50.npy",
#             "datafile/AbaqusOutput_Job-Example-60.npy",
#             "datafile/AbaqusOutput_Job-Example-70.npy",
#             "datafile/AbaqusOutput_Job-Example-80.npy",
#             ]
# force = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]

datafiles = ["datafile/AbaqusOutput_Job-Example-80.npy",
            ]
force = [0.08]

for id, path in enumerate(datafiles):
    ## true solution
    dataset_e = np.load(path)
    x_e = dataset_e[:, 1:3]
    u_e = dataset_e[:, 3:5]
    sigma_e = dataset_e[:, 10:14]

    fx_fem_e = Rbf(x_e[:, 0], x_e[:, 1], u_e[:, 0])
    fy_fem_e = Rbf(x_e[:, 0], x_e[:, 1], u_e[:, 1])
    # ux_fem_e = fx_fem_e(mesh_coord[:, 0], mesh_coord[:, 1])[:, None]
    # uy_fem_e = fy_fem_e(mesh_coord[:, 0], mesh_coord[:, 1])[:, None]
    # u_fem_e = np.concatenate((ux_fem_e, uy_fem_e), axis=1 )

    f11_e = Rbf(x_e[:, 0], x_e[:, 1], sigma_e[:, 0])
    f22_e = Rbf(x_e[:, 0], x_e[:, 1], sigma_e[:, 1])
    f12_e = Rbf(x_e[:, 0], x_e[:, 1], sigma_e[:, 3])
    s11_e_intf = f11_e(x_intf[:, 0], x_intf[:, 1])
    s22_e_intf = f22_e(x_intf[:, 0], x_intf[:, 1])
    s12_e_intf = f12_e(x_intf[:, 0], x_intf[:, 1])

    nx = x_intf[:, 0]/r0
    ny = x_intf[:, 1]/r0
    tx = s11_e_intf*nx+s12_e_intf*ny
    ty = s12_e_intf*nx+s22_e_intf*ny
    t_e = np.concatenate((tx[:, None], ty[:, None]), axis=1) # point toward outside

    ##
    ux_fem_intf = fx_fem_e(x_intf[:, 0], x_intf[:, 1])[:, None]
    uy_fem_intf = fy_fem_e(x_intf[:, 0], x_intf[:, 1])[:, None]
    u_fem_intf = np.concatenate((ux_fem_intf, uy_fem_intf), axis=1 )*U_scaling

    x_b = u_fem_intf.reshape((1, -1), order='F')
    x_b = np.tile(x_b, (dim_intf, 1))
    x_b_tensor = torch.tensor(x_b, dtype=torch.float).to(device)

    tx_intf = to_numpy(model_tx.forward(x_b_tensor, x_t_tensor))        ## tx
    ty_intf = to_numpy(model_ty.forward(x_b_tensor, x_t_tensor))        ## ty

    ##
    fig = plt.figure(constrained_layout=False, figsize=(15, 5))
    gs = fig.add_gridspec(1, 3)
    ax = fig.add_subplot(gs[0])
    ax.plot(tx_intf, color='blue', label='DeepONet')
    ax.plot(T_scaling*tx, color='red', label='FEM')
    # ax.set_ylim([-1, 1])

    ax = fig.add_subplot(gs[1])
    ax.plot(ty_intf, color='blue', label='DeepONet')
    ax.plot(T_scaling*ty, color='red', label='FEM')
    # ax.set_ylim([-1, 1])

    ax.set_xlabel('# iter')
    # ax.set_ylabel('$L_{2}$ error')
    # ax.set_yscale('log')
    ax.legend()
    plt.savefig("test.png")
    plt.close()

    exit()




