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
import os

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
niter = 40
theta = 0.5
r0 = 0.3

U_scaling = 100
S_scaling = 100

########## DeepONet ##########
device = 'cuda'
dim_br = [80, 100, 100, 100]
dim_tr = [2, 40, 40, 100]

## initialization
model_s11 = opnn(dim_br, dim_tr).to(device)
model_s11 = model_s11.float()
s11_path = "./s11_model/model_chkpts.pt"
checkpoint = torch.load(s11_path)
model_s11.load_state_dict(checkpoint["model_state_dict"])

model_s12 = opnn(dim_br, dim_tr).to(device)
model_s12 = model_s12.float()
s12_path = "./s12_model/model_chkpts.pt"
checkpoint = torch.load(s12_path)
model_s12.load_state_dict(checkpoint["model_state_dict"])

model_s22 = opnn(dim_br, dim_tr).to(device)
model_s22 = model_s22.float()
s22_path = "./s22_model/model_chkpts.pt"
checkpoint = torch.load(s22_path)
model_s22.load_state_dict(checkpoint["model_state_dict"])

## interface and lines
x_intf = np.loadtxt("x_intf_ext.dat")
n_intf = normal_unit(x_intf) ## assume circle
x_t_tensor = torch.tensor(x_intf, dtype=torch.float).to(device) ## tensorize the input of the trunk net
dim_intf = x_intf.shape[0]

##
# x = np.linspace(-1, 1, 101)[:, None]
# tmp = np.ones((101, 1))
# x_top = np.concatenate((x, tmp), axis=1)
# x_right = np.concatenate((tmp, x[::-1]), axis=1)
# x_bot = np.concatenate((x[::-1], -1*tmp), axis=1)
# x_left = np.concatenate((-1*tmp, x), axis=1)
# x_bc_out = np.concatenate((x_top, x_right, x_bot, x_left), axis=0)
# np.savetxt("x_bc_out.dat", x_bc_out)

x_bc_out = np.loadtxt("x_bc_out.dat")
x_bc_in = np.concatenate((x_intf, x_intf[0:1]), axis=0)

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
    dump_outline = "outline_f" + str(force[id])
    os.makedirs(dump_outline, exist_ok=True)

    ## Initialization
    u = TrialFunction(V)
    d = u.geometric_dimension()
    v = TestFunction(V)
    f = Constant((0, 0))
    T = Constant((0, force[id]))
    T_ini = Constant((0.1, 0))
    a = inner(sigma(u), epsilon(v))*dx
    L = dot(f, v)*dx + dot(T, v)*ds_fem(1) + dot(T_ini, v)*ds_fem(2) ## ds is on the boundary

    ## true solution
    dataset_e = np.load(path)
    x_e = dataset_e[:, 1:3]
    u_e = dataset_e[:, 3:5]
    sigma_e = dataset_e[:, 10:14]

    fx_fem_e = Rbf(x_e[:, 0], x_e[:, 1], u_e[:, 0])
    fy_fem_e = Rbf(x_e[:, 0], x_e[:, 1], u_e[:, 1])
    ux_fem_e = fx_fem_e(mesh_coord[:, 0], mesh_coord[:, 1])[:, None]
    uy_fem_e = fy_fem_e(mesh_coord[:, 0], mesh_coord[:, 1])[:, None]
    u_fem_e = np.concatenate((ux_fem_e, uy_fem_e), axis=1 )

    f11_e = Rbf(x_e[:, 0], x_e[:, 1], sigma_e[:, 0])
    f22_e = Rbf(x_e[:, 0], x_e[:, 1], sigma_e[:, 1])
    f12_e = Rbf(x_e[:, 0], x_e[:, 1], sigma_e[:, 3])
    s11_e_intf = f11_e(x_intf[:, 0], x_intf[:, 1])
    s22_e_intf = f22_e(x_intf[:, 0], x_intf[:, 1])
    s12_e_intf = f12_e(x_intf[:, 0], x_intf[:, 1])

    s11_e_mesh = f11_e(mesh_coord[:, 0], mesh_coord[:, 1])
    s12_e_mesh = f12_e(mesh_coord[:, 0], mesh_coord[:, 1])
    s22_e_mesh = f22_e(mesh_coord[:, 0], mesh_coord[:, 1])

    nx = x_intf[:, 0]/r0
    ny = x_intf[:, 1]/r0
    tx = s11_e_intf*nx+s12_e_intf*ny
    ty = s12_e_intf*nx+s22_e_intf*ny
    t_e = np.concatenate((tx[:, None], ty[:, None]), axis=1) # point toward outside

    ## dump true solution
    ux_out_e = fx_fem_e(x_bc_out[:, 0], x_bc_out[:, 1])[:, None]
    uy_out_e = fy_fem_e(x_bc_out[:, 0], x_bc_out[:, 1])[:, None]
    u_out_e = np.concatenate((ux_out_e, uy_out_e), axis=1 )

    ux_in_e = fx_fem_e(x_bc_in[:, 0], x_bc_in[:, 1])[:, None]
    uy_in_e = fy_fem_e(x_bc_in[:, 0], x_bc_in[:, 1])[:, None]
    u_in_e = np.concatenate((ux_in_e, uy_in_e), axis=1 )

    f_out = dump_outline + "/f_out_e.txt"
    f_in = dump_outline + "/f_in_e.txt"
    u_out_e_cat = np.concatenate((u_out_e, x_bc_out), axis=1)
    u_in_e_cat = np.concatenate((u_in_e, x_bc_in), axis=1)
    np.savetxt(f_out, u_out_e_cat)
    np.savetxt(f_in, u_in_e_cat)

    ########## Multiscale Coupling ##########
    err_his = []
    for iter in range(0, niter):
        ## FEM 
        if iter == 0:
            u = Function(V)
            solve(a == L, u, bc_bot)
            
        else:
            sigma_out = projection(project(sigma(u), W), x_out).reshape(-1, 2, 2)     ## stress from FEM
            t_out = np.matmul(sigma_out, n_out[:, :, None])[:, :, 0]                    ## n_intf pointing toward outside   

            t1_fem = np.concatenate((x_out, -t_out), axis=1)        ## need to turn t_out input pointing inward
            t2_fem = np.concatenate((x_intf, -lmd/S_scaling), axis=1) ## need to rescale, pointing toward inside
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
        u_intf = projection(u, x_intf)             ## interfacial displacement from FEM
        u_intf = u_intf*U_scaling
        
        x_b = u_intf.reshape((1, -1), order='F')
        x_b = np.tile(x_b, (dim_intf, 1))
        x_b_tensor = torch.tensor(x_b, dtype=torch.float).to(device)

        s11_intf = to_numpy(model_s11.forward(x_b_tensor, x_t_tensor))        ## sigma 11
        s12_intf = to_numpy(model_s12.forward(x_b_tensor, x_t_tensor))        ## sigma 12
        s22_intf = to_numpy(model_s22.forward(x_b_tensor, x_t_tensor))        ## sigma 22

        tx_intf = s11_intf*nx[:, None] + s12_intf*ny[:, None]
        ty_intf = s12_intf*nx[:, None] + s22_intf*ny[:, None]
        t_intf = np.concatenate((tx_intf, ty_intf), axis=1)

        ## traction on the interface from FEM
        sigma_intf = projection(project(sigma(u), W), x_intf).reshape(-1, 2, 2)     ## stress from FEM
        lmd = np.matmul(sigma_intf, n_intf[:, :, None])[:, :, 0]                    ## n_intf pointing toward outside
        lmd = lmd*S_scaling                                                               ## scaling!! lmd is traction

        ## dynamic theta
        # if iter == 0:
        #     s1_old = projection(project(sigma(u), W), x_intf)
        #     s2_old = np.concatenate((s11_intf, s12_intf, s12_intf, s22_intf), axis=1)/S_scaling
        #     # s1_old = projection(project(sigma(u), W), x_intf)[:, 0:1]
        #     # s2_old = s11_intf/S_scaling
        # else:
        #     s1_new = projection(project(sigma(u), W), x_intf)
        #     s2_new = np.concatenate((s11_intf, s12_intf, s12_intf, s22_intf), axis=1)/S_scaling
        #     # s1_new = projection(project(sigma(u), W), x_intf)[:, 0:1]
        #     # s2_new = s11_intf/S_scaling
        #     e1 = s1_new - s1_old
        #     e2 = s2_new - s2_old
        #     s1_old = np.copy(s1_new)
        #     s2_old = np.copy(s2_new)

        #     theta = (e1 * (e1 - e2)).sum()/((e1 - e2)**2).sum()
        #     print(theta)

        ## update lambda (traction in x, y direction)
        lmd = theta*t_intf + (1 - theta)*lmd

        ## error 
        s_fem_pred = projection(project(sigma(u), W), mesh_coord)

        err_1 = ((s_fem_pred[:, 0] - s11_e_mesh)**2).mean() + \
                ((s_fem_pred[:, 1] - s12_e_mesh)**2).mean() + \
                ((s_fem_pred[:, 3] - s22_e_mesh)**2).mean()
        err_2 = ((lmd/S_scaling - t_e)**2).mean()

        rela_err_1 = (abs((s_fem_pred[:, 0] - s11_e_mesh)/s11_e_mesh.max()) + \
                abs((s_fem_pred[:, 1] - s12_e_mesh)/s12_e_mesh.max()) + \
                abs((s_fem_pred[:, 3] - s22_e_mesh)/s22_e_mesh.max()) ).mean()*1/3
        rela_err_2 = abs((lmd/S_scaling - t_e)/t_e.max()).mean()

        # print( (s_fem_pred[:, 0] - s11_e_mesh)/s11_e_mesh )
        

        err_his.append( np.array([err_1, err_2, rela_err_1, rela_err_2]) )
        print(f'iter: {iter}, err_1: {err_1:3g}, err_2: {err_2:3g}, rela_err_1: {rela_err_1}, rela_err_2: {rela_err_2}')

        ## save outlines
        u_bc_out = projection(u, x_bc_out)
        u_bc_in = projection(u, x_bc_in)
        f_out = dump_outline + "/iter_" + str(iter) + "_f_out.txt"
        f_in = dump_outline + "/iter_" + str(iter) + "_f_in.txt"

        u_bc_out_cat = np.concatenate((u_bc_out, x_bc_out), axis=1)
        u_bc_in_cat = np.concatenate((u_bc_in, x_bc_in), axis=1)
        np.savetxt(f_out, u_bc_out_cat)
        np.savetxt(f_in, u_bc_in_cat)

        # if err_1 <= eps and err_2 <= eps:
        #     break

    err_his = np.array(err_his)
    np.savetxt("err_his.dat", err_his)

########## Save and Plot ##########
## save FEM
u_fem_pred = projection(u, mesh_coord)
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
# ax.legend()
plt.savefig("disp_stress.png")
plt.close()

