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

def get_true_sol():
    square_out = Rectangle(Point(0,0), Point(1.0, 1.0))
    mesh_e = generate_mesh(square_out, 64)

    W_e = TensorFunctionSpace(mesh_e, "P", 2)
    V_e = VectorFunctionSpace(mesh_e, "P", 2)
    bc_e_bot = DirichletBC(V_e, Constant((0, 0)), clamped_boundary)

    boundaries = MeshFunction("size_t", mesh_e, mesh_e.topology().dim() - 1)
    boundaries.set_all(0)

    top = AutoSubDomain(lambda x: near(x[1], 1))
    top.mark(boundaries, 1)
    ds_e = ds(subdomain_data=boundaries)

    u_e = TrialFunction(V_e)
    v_e = TestFunction(V_e)
    f_e = Constant((0, 0))
    T_e = Constant((0, force))
    a_e = inner(sigma(u_e), epsilon(v_e))*dx
    L_e = dot(f_e, v_e)*dx + dot(T_e, v_e)*ds_e(1)

    u_e = Function(V_e)
    solve(a_e == L_e, u_e, bc_e_bot)

    sigma_u_e = project(sigma(u_e), W_e)
    return u_e

## parameters
mu = 1
rho = 1
lambda_ = 1.25
force = 0.1
dim = 11
eps = 1e-10
niter = 50
theta = 0.25

## mesh lower
square1 = Rectangle(Point(0.0, 0.0), Point(1.0, 0.5))
mesh1 = generate_mesh(square1, 32)
x1 = mesh1.coordinates()

## mesh upper
square2 = Rectangle(Point(0.0, 0.5), Point(1.0, 1.0))
mesh2 = generate_mesh(square2, 32)
x2 = mesh2.coordinates()

## interface and lines
x_h = np.linspace(0.0, 1.0, dim)[:, None]
ones = np.ones((dim, 1))
x_intf = np.concatenate((x_h, ones*0.5), axis=1)
x1_bot = np.concatenate((x_h, ones*0.0), axis=1)

xx, yy = np.meshgrid(x_h, np.linspace(0, 0.45, dim)[:, None])
x1_inner = np.concatenate((xx.flatten()[:, None], yy.flatten()[:, None] ), axis=1)

xx, yy = np.meshgrid(x_h, np.linspace(0.55, 1.0, dim)[:, None])
x2_inner = np.concatenate((xx.flatten()[:, None], yy.flatten()[:, None] ), axis=1)

## lower section
W1 = TensorFunctionSpace(mesh1, "P", 2)
V1 = VectorFunctionSpace(mesh1, "P", 2)

boundaries1 = MeshFunction("size_t", mesh1, mesh1.topology().dim() - 1)
boundaries1.set_all(0)
intf = AutoSubDomain(lambda x: near(x[1], 0.5))
intf.mark(boundaries1, 1)
ds1 = ds(subdomain_data=boundaries1)

def clamped_boundary(x, on_boundary):
    return on_boundary and x[1] < eps
bc1_bot = DirichletBC(V1, Constant((0, 0)), clamped_boundary)

## model up
W2 = TensorFunctionSpace(mesh2, "P", 2)
V2 = VectorFunctionSpace(mesh2, "P", 2)

boundaries2 = MeshFunction("size_t", mesh2, mesh2.topology().dim() - 1)
boundaries2.set_all(0)
top = AutoSubDomain(lambda x: near(x[1], 1.0))
top.mark(boundaries2, 1)
ds2 = ds(subdomain_data=boundaries2)

def intf_boundary(x, on_boundary):
    return on_boundary and abs(x[1]-0.5) < eps
bc2_intf = DirichletBC(V2, ("0", "0"), intf_boundary)

u2 = TrialFunction(V2)
d = u2.geometric_dimension()
v2 = TestFunction(V2)
f2 = Constant((0, 0))
T2 = Constant((0, force)) ## traction on the top surface
a2 = inner(sigma(u2), epsilon(v2))*dx
L2 = dot(f2, v2)*dx + dot(T2, v2)*ds2(1) ## ds is on the boundary

## true solution
u_e = get_true_sol()
u_e1 = projection(u_e, x1)
u_e2 = projection(u_e, x2)

n_intf = np.array([[0, 0], [-1, 0], [0, 0], [0, -1]])

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

    # print(sigma1_intf, sigma1_intf_1)
    # exit()

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