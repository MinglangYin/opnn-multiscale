"""
Minglang Yin, minglang_yin@brown.edu
Coupling DeepONet with Linear Finite Element Model
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
force = 0.08
eps = 1e-6
niter = 5
theta = 0.5
r0 = 0.3

########## DeepONet ##########
device = 'cuda'
dim_br = [80, 100, 100, 100]
dim_tr = [2, 100, 100, 100]

## initialization
model = opnn(dim_br, dim_tr).to(device)
model = model.float()

opnn_path = "./CheckPts/model_chkpts.pt"
checkpoint = torch.load(opnn_path)
model.load_state_dict(checkpoint["model_state_dict"])

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

## Initialization
u = TrialFunction(V)
d = u.geometric_dimension()
v = TestFunction(V)
f = Constant((0, 0))
T = Constant((0, force))
a = inner(sigma(u), epsilon(v))*dx
L = dot(f, v)*dx + dot(T, v)*ds_fem(1) ## ds is on the boundary

## true solution
dataset_e = np.load("datafile/AbaqusOutput_Job-Example_PlaneStrainPlasticity.npy")
x_e = dataset_e[:, 1:3]
u_e = dataset_e[:, 3:5]
sigma_e = dataset_e[:, 10:14]

fx_fem_e = Rbf(x_e[:, 0], x_e[:, 1], u_e[:, 0])
fy_fem_e = Rbf(x_e[:, 0], x_e[:, 1], u_e[:, 1])
ux_fem_e = fx_fem_e(mesh_coord[:, 0], mesh_coord[:, 1])[:, None]
uy_fem_e = fy_fem_e(mesh_coord[:, 0], mesh_coord[:, 1])[:, None]
u_fem_e = np.concatenate((ux_fem_e, uy_fem_e), axis=1 )

fxx_e = Rbf(x_e[:, 0], x_e[:, 1], sigma_e[:, 0])
fyy_e = Rbf(x_e[:, 0], x_e[:, 1], sigma_e[:, 1])
fxy_e = Rbf(x_e[:, 0], x_e[:, 1], sigma_e[:, 3])
sxx_e_intf = fxx_e(x_intf[:, 0], x_intf[:, 1])
syy_e_intf = fyy_e(x_intf[:, 0], x_intf[:, 1])
sxy_e_intf = fxy_e(x_intf[:, 0], x_intf[:, 1])

nx = x_intf[:, 0]/r0
ny = x_intf[:, 1]/r0
tx = sxx_e_intf*nx+sxy_e_intf*ny
ty = sxy_e_intf*nx+syy_e_intf*ny
t_e = np.concatenate((tx[:, None], ty[:, None]), axis=1) # point toward outside

########## Multiscale Coupling ##########
err_his = []
dyn_theta = 0

t_intf_prev = 0
lmd_prev = 0
for iter in range(0, niter):
    ## FEM 
    if iter == 0:
        u = Function(V)
        solve(a == L, u, bc_bot)
        
    else:
        sigma_out = projection(project(sigma(u), W), x_out).reshape(-1, 2, 2)     ## stress from FEM
        t_out = np.matmul(sigma_out, n_out[:, :, None])[:, :, 0]                    ## n_intf pointing toward outside   

        t1_fem = np.concatenate((x_out, -t_out), axis=1)
        t2_fem = np.concatenate((x_intf, -lmd/100), axis=1) ## need to rescale, pointing toward inside
        T_fit = np.concatenate((t1_fem, t2_fem), axis=0) 
        T_exp = data2exp_2d(T_fit, V)

        ## 
        u = TrialFunction(V)
        v = TestFunction(V)
        a = inner(sigma(u), epsilon(v))*dx
        L = dot(f, v)*dx + dot(T, v)*ds_fem(1) + dot(T_exp, v)*ds_fem(2)

        u = Function(V)
        solve(a == L, u, bc_bot)

    ## DeepONet
    u_intf = projection(u, x_intf)*100             ## interfacial displacement from FEM
    
    x_b = u_intf.reshape((1, -1), order='F')
    x_b = np.tile(x_b, (dim_intf, 1))
    x_b_tensor = torch.tensor(x_b, dtype=torch.float).to(device)
    t_intf = to_numpy(model.forward(x_b_tensor, x_t_tensor))        ## traction, pointing toward outside
    
    ## traction on the interface from FEM
    sigma_intf = projection(project(sigma(u), W), x_intf).reshape(-1, 2, 2)     ## stress from FEM
    lmd = np.matmul(sigma_intf, n_intf[:, :, None])[:, :, 0]                    ## n_intf pointing toward outside
    lmd = lmd*100                                                               ## scaling!!

    ## dynamic theta
    # if iter == 0:
    #     u_in_old = projection(u_in, x_intf)
    #     u_out_old = projection(u_out, x_intf)
    # else:
    #     u_in_new = projection(u_in, x_intf)
    #     u_out_new = projection(u_out, x_intf)
    #     e1 = u_in_new - u_in_old
    #     e2 = u_out_new - u_out_old
    #     u_in_old = np.copy(u_in_new)
    #     u_out_old = np.copy(u_out_new)
    #     theta = (e1 * (e1 - e2)).sum()/((e1 - e2)**2).sum()

    ## update lambda (traction in x, y direction)
    # print(t_intf)
    # print(lmd)
    lmd = theta*t_intf + (1 - theta)*lmd

    ## error 
    u_fem_pred = projection(u, mesh_coord)
    
    # tmp = u_fem_pred - u_fem_e
    # for diff in tmp:
    #     print(diff)

    # print(t_e-lmd/100)
    # exit()
    
    err_1 = ((u_fem_pred - u_fem_e)**2).mean()
    err_2 = ((lmd/100 - t_e)**2).mean()
    err_his.append( np.array([err_1, err_2]) )
    print(f'iter: {iter}, err_1: {err_1:3g}, err_2: {err_2:3g}')

    # if err_1 <= eps and err_2 <= eps:
    #     break

err_his = np.array(err_his)

########## Save and Plot ##########
## save FEM
sigma = project(sigma(u), W)

file_results = XDMFFile("FEM.xdmf")
file_results.parameters["flush_output"] = True
file_results.parameters["functions_share_mesh"] = True
file_results.write(u, 0.)
file_results.write(sigma, 0.)

## figures
fig = plt.figure(constrained_layout=False, figsize=(5, 5))
gs = fig.add_gridspec(1, 1)
ax = fig.add_subplot(gs[0])
ax.plot(err_his[:, 0], color='blue', label='FEM')
ax.plot(err_his[:, 1], color='red', label='DeepONet')
# ax.axhline(y=eps, linestyle='--', color='black',  label='$\epsilon$')

ax.set_xlabel('# iter')
ax.set_ylabel('$L_{2}$ error')
ax.set_yscale('log')
ax.legend()
plt.savefig("convergence.png")
plt.close()

## scatter and line plot
fig = plt.figure(constrained_layout=False, figsize=(10, 14))
gs = fig.add_gridspec(3, 2)
ax = fig.add_subplot(gs[0])
ax.plot(t_e[:, 0]*100, color='blue', label='True Tx')
ax.plot(lmd[:, 0], color='red', linestyle='--', label='DeepONet Tx')
ax.set_xlabel("Points ID")
ax.legend()

ax = fig.add_subplot(gs[1])
ax.plot(t_e[:, 1]*100, color='blue', label='True Ty')
ax.plot(lmd[:, 1], color='red', linestyle='--', label='DeepONet Ty')
ax.set_xlabel("Points ID")
ax.legend()

ax = fig.add_subplot(gs[2])
h = ax.scatter(mesh_coord[:, 0], mesh_coord[:, 1], c = u_fem_e[:, 0]*100)
h.set_clim(vmin=-4, vmax=4)
ax.set_title("true ux")
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, ax=ax, cax=cax)

ax = fig.add_subplot(gs[3])
h = ax.scatter(mesh_coord[:, 0], mesh_coord[:, 1], c = u_fem_pred[:, 0]*100)
h.set_clim(vmin=-4, vmax=4)
ax.set_title("FEM ux")
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, ax=ax, cax=cax)

ax = fig.add_subplot(gs[4])
h = ax.scatter(mesh_coord[:, 0], mesh_coord[:, 1], c = u_fem_e[:, 1]*100)
# h.set_clim(vmin=-4, vmax=4)
ax.set_title("true uy")
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, ax=ax, cax=cax)

ax = fig.add_subplot(gs[5])
h = ax.scatter(mesh_coord[:, 0], mesh_coord[:, 1], c = u_fem_pred[:, 1]*100)
# h.set_clim(vmin=-4, vmax=4)
ax.set_title("FEM uy")
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, ax=ax, cax=cax)

# ax.set_xlabel('# iter')
# ax.set_ylabel('$L_{2}$ error')
# ax.set_yscale('log')
ax.legend()
plt.savefig("disp_stress.png")
plt.close()

### save for plot outline

u_intf = projection(u, x_intf)
np.savetxt("pred_in.dat", x_intf+u_intf)
