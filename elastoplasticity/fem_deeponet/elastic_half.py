"""
Minglang Yin, minglang_yin@brown.edu
Coupling DeepONet with Linear Finite Element Model
"""

import numpy as np
import matplotlib.pyplot as plt

from fenics import *
from ufl import nabla_div
from mshr import *
from scipy.interpolate import interp2d

def epsilon(u):
    return 0.5*(nabla_grad(u)+nabla_grad(u).T)

def sigma(u):
    return lambda_*nabla_div(u)*Identity(d) + 2*mu*epsilon(u)

def projection(u, pts):
    u_prj = []
    for x in pts:
        u_prj.append(u(x))
    return np.asarray(u_prj)

def data2exp_2d(data, V):
    x, y, values = data[:,0], data[:,1], data[:,2:]
    interpolant1 = interp2d(x, y, values[:, 0:1], kind='cubic', copy=False, bounds_error=True)
    interpolant2 = interp2d(x, y, values[:, 1:2], kind='cubic', copy=False, bounds_error=True)

    class ExpressionFromScipyFunction(UserExpression):
        def __init__(self, f, **kwargs):
            self._f1 = f[0]
            self._f2 = f[1]
            UserExpression.__init__(self, **kwargs)
        def eval(self, values, x):
            values[0] = self._f1(*x)
            values[1] = self._f2(*x)

    u = ExpressionFromScipyFunction((interpolant1, interpolant2), element=V.ufl_element())
    return u

## parameters
mu = 1
rho = 1
lambda_ = 1.25
force = 0.1
dim = 11
eps = 1e-10
niter = 50
theta = 0.25

## mesh
square = Rectangle(Point(-1.0, -1.0), Point(1.0, 1.0))
circle = Circle(Point(0.0, 0.0), 0.3)
mesh = generate_mesh(square-circle, 32)
mesh_coord = mesh.coordinates()

## interface and lines


## FEM
W = TensorFunctionSpace(mesh, "P", 2)
V = VectorFunctionSpace(mesh, "P", 2)

boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundaries.set_all(0)
#intf = AutoSubDomain(lambda x: near(x[1], 0.5))
intf.mark(boundaries, 1)
ds_fem = ds(subdomain_data=boundaries)

def clamped_boundary(x, on_boundary):
    return on_boundary and x[1] < eps ## wrong!!!
bc_bot = DirichletBC(V, Constant((0, 0)), clamped_boundary)

## iteration
err_his = []
dyn_theta = 0
for iter in range(0, niter):
    if iter == 0:
        ## model (up)
        u2 = Function(V2)
        solve(a2 == L2, u2, [bc2_intf])

    else:
        ## model (up)
        u2_zeros = np.zeros_like(x2_inner)
        ux2_inner = np.concatenate((x2_inner, u2_zeros), axis=1)
        ux2_intf = np.concatenate((x_intf, lmd), axis=1)
        ux2_tmp = np.concatenate((ux2_inner, ux2_intf), axis=0)

        bc2_exp = data2exp_2d(ux2_tmp, V2)
        bc2_intf = DirichletBC(V2, bc2_exp, intf_boundary)

        u2 = TrialFunction(V2)
        v2 = TestFunction(V2)
        f2 = Constant((0, 0))
        T2 = Constant((0, force))
        a2 = inner(sigma(u2), epsilon(v2))*dx
        L2 = dot(f2, v2)*dx + dot(T2, v2)*ds2(1) ## ds is on the boundary

        u2 = Function(V2)
        solve(a2 == L2, u2, [bc2_intf])

    ## model (lower)
    sigma2 = project(sigma(u2), W2)
    sigma1_intf = -np.matmul(projection(sigma2, x_intf), n_intf)
    ux_intf = np.concatenate((x_intf, sigma1_intf), axis=1)

    # sigma1 = project(sigma(u1), W1)
    # sigma1_inner = np.matmul(projection(sigma1, x1_inner), n_intf)
    # ux_inner = np.concatenate((x1_inner, sigma1_inner), axis=1)

    u1_inner = np.zeros_like(x1_inner)
    ux_inner = np.concatenate((x1_inner, u1_inner), axis=1)

    ux_tb = np.concatenate((ux_intf, ux_inner), axis=0)
    f_top = data2exp_2d(ux_tb, V1)

    f1 = Constant((0, 0))
    u1 = TrialFunction(V1)
    v1 = TestFunction(V1)
    a1 = inner(sigma(u1), epsilon(v1))*dx
    L1 = dot(f1, v1)*dx + dot(f_top, v1)*ds1(1)

    u1 = Function(V1)
    solve(a1 == L1, u1, bc1_bot)

    ## stress from 
    sigma1 = project(sigma(u1), W1)
    sigma1_intf_1 = np.matmul(projection(sigma1, x_intf), n_intf)

    ## dynamic theta
    if iter == 0:
        u_in_old = projection(u_in, x_intf)
        u_out_old = projection(u_out, x_intf)
    else:
        u_in_new = projection(u_in, x_intf)
        u_out_new = projection(u_out, x_intf)
        e1 = u_in_new - u_in_old
        e2 = u_out_new - u_out_old
        u_in_old = np.copy(u_in_new)
        u_out_old = np.copy(u_out_new)

        theta = (e1 * (e1 - e2)).sum()/((e1 - e2)**2).sum()

    ## lambda
    u1_intf = projection(u1, x_intf)
    lmd = projection(u2, x_intf)
    lmd = theta*u1_intf + (1 - theta)*lmd

    ## error 
    u1_pred = projection(u1, x1)
    u2_pred = projection(u2, x2)
    err_1 = ((u_e1 - u1_pred)**2).mean()
    err_2 = ((u_e2 - u2_pred)**2).mean()
    err_his.append( np.array([err_1, err_2]) )
    print(f'iter: {iter}, err_in: {err_1:3g}, err_2: {err_2:3g}')

    if err_in <= eps and err_out <= eps:
        break

err_his = np.array(err_his)

##
sigma_u1 = project(sigma(u1), W1)
sigma_u2 = project(sigma(u2), W2)

file_results = XDMFFile("box_down.xdmf")
file_results.parameters["flush_output"] = True
file_results.parameters["functions_share_mesh"] = True
file_results.write(u1, 0.)
file_results.write(sigma_u1, 0.)

file_results = XDMFFile("box_up.xdmf")
file_results.parameters["flush_output"] = True
file_results.parameters["functions_share_mesh"] = True
file_results.write(u2, 0.)
file_results.write(sigma_u2, 0.)

##
fig = plt.figure(constrained_layout=False, figsize=(4, 4))
gs = fig.add_gridspec(1, 1)
ax = fig.add_subplot(gs[0])
ax.plot(err_his[:, 0], color='blue', label='$u1$ (FEM1)')
ax.plot(err_his[:, 1], color='red', label='$u2$ (FEM2)')
# ax.axhline(y=eps, linestyle='--', color='black',  label='$\epsilon$')

ax.set_xlabel('# iter')
ax.set_ylabel('$L_{2}$ error')
ax.set_yscale('log')
ax.legend()
plt.savefig("convergence.png")
plt.close()