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
            self._f = f
            UserExpression.__init__(self, **kwargs)
        def eval(self, values, x):
            values[:] = self._f(*x)

    u = ExpressionFromScipyFunction((interpolant1, interpolant2), element=V.ufl_element())
    return u

def get_true_sol():
    square_out = Rectangle(Point(0,0), Point(1.0, 1.0))
    mesh_e = generate_mesh(square_out, 32)

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
    return u_e

## parameters
mu = 1
rho = 1
h = 0.4
lambda_ = 1.25
tol = 1e-14
g = 9.8
force = 0.2
dim = 11

## mesh
square_out = Rectangle(Point(0,0), Point(1.0, 1.0))
square_in = Rectangle(Point(0.3,0.3), Point(0.7, 0.7))
mesh_out = generate_mesh(square_out - square_in, 32)
mesh_in = generate_mesh(square_in, 32)

x_out = mesh_out.coordinates()
x_in = mesh_in.coordinates()

x_lin = np.linspace(0.3, 0.7, dim)[:, None]
ones = np.ones((dim, 1))
x_intf_top = np.concatenate((x_lin, ones*0.7), axis=1)
x_intf_bot = np.concatenate((x_lin, ones*0.3), axis=1)
x_intf_left = np.concatenate((ones*0.3, x_lin), axis=1)
x_intf_right = np.concatenate((ones*0.7, x_lin), axis=1)
x_intf = np.concatenate((x_intf_top, x_intf_bot, x_intf_left, x_intf_right), axis=0)

## model out
W_out = TensorFunctionSpace(mesh_out, "P", 2)
V_out = VectorFunctionSpace(mesh_out, "P", 2)
def clamped_boundary(x, on_boundary):
    return on_boundary and x[1]<tol
bc_out_bot = DirichletBC(V_out, Constant((0, 0)), clamped_boundary)

boundaries = MeshFunction("size_t", mesh_out, mesh_out.topology().dim() - 1)
boundaries.set_all(0)

top = AutoSubDomain(lambda x: near(x[1], 1))
top.mark(boundaries, 1)
ds = ds(subdomain_data=boundaries)

u_out = TrialFunction(V_out)
d = u_out.geometric_dimension()
v_out = TestFunction(V_out)
f_out = Constant((0, 0))
T_out = Constant((0, force))
a_out = inner(sigma(u_out), epsilon(v_out))*dx
L_out = dot(f_out, v_out)*dx + dot(T_out, v_out)*ds(1) ## ds is on the boundary

## model in
u_init = Expression( ("0.0","0.0"), degree=1)
W_in = TensorFunctionSpace(mesh_in, "P", 2)
V_in = VectorFunctionSpace(mesh_in, "P", 2)
def in_boundary(x, on_boundary):
    return on_boundary and (abs(x[0] - 0.3) < DOLFIN_EPS or abs(x[0] - 0.7) < DOLFIN_EPS \
            or abs(x[1] - 0.3) < DOLFIN_EPS or abs(x[1] - 0.7) < DOLFIN_EPS)
bc_in = DirichletBC(V_in, u_init, in_boundary)

u_in = TrialFunction(V_in)
v_in = TestFunction(V_in)
f_in = Constant((0, 0))
a_in = inner(sigma(u_in), epsilon(v_in))*dx
L_in = dot(f_in, v_in)*dx

## true solution
u_e = get_true_sol()
u_e_out = projection(u_e, x_out)
u_e_in = projection(u_e, x_in)

## inner boundary displacement
# def inner_boundary(x, on_boundary):
#     return on_boundary and abs(x[0] - 0.3)<tol or abs(x[0] - 0.7)<tol or abs(x[1] - 0.3)<tol or abs(x[1] - 0.7)<tol

eps = 1e-10
niter = 2
lmd = 0.0
theta = 0.25
err_his = []
dyn_theta = 0
for iter in range(0, niter):
    if iter == 0:
        ## model(out)
        u_out = Function(V_out)
        solve(a_out == L_out, u_out, bc_out_bot)
        
        ## model(in)
        u_in = Function(V_in)
        solve(a_in == L_in, u_in, bc_in)

    else:
        ## model(out)
        bc_out_in = data2exp_2d(np.concatenate((x_intf, lmd), axis=1), V_out)
        exit()
        bc_out_inner = DirichletBC(V_out, (bc_u, bc_v), in_boundary)
        u_out = solve(a_out == L_out, u_out, [bc_out_bot, bc_out_inner])
        exit()
    
        ## model(in)
        # u_out = solve
        u1_x = project(u1.dx(0), V1)
        u_intf_x = u1_x(x_intf)

    ## dynamic theta
    # if iter == 0:
    #     u1_old = u1(x_intf)
    #     u2_old = u2(x_intf)
    # else:
    #     u1_new = u1(x_intf)
    #     u2_new = u2(x_intf)
    #     e1 = u1_new - u1_old
    #     e2 = u2_new - u2_old
    #     theta = (e1 * (e1 - e2)).sum()/((e1 - e2)**2).sum()

    ## lambda
    u_in_intf = projection(u_in, x_intf)
    lmd = projection(u_out, x_intf)
    lmd = theta*u_in_intf + (1 - theta)*lmd

    ## error 
    u_in_pred = projection(u_in, x_in)
    u_out_pred = projection(u_out, x_out)
    err_in = ((u_e_in - u_in_pred)**2).mean()
    err_out = ((u_e_out - u_out_pred)**2).mean()
    err_his.append( np.array([err_in, err_out]) )
    print(f'iter: {iter}, err_in: {err_in:3g}, err_2: {err_out:3g}')

    # if err_in <= eps and err_out <= eps:
    #     break

err_his = np.array(err_his)


##
sigma_u_out = project(sigma(u_out), W_out)
sigma_u_in = project(sigma(u_in), W_in)

file_results = XDMFFile("box_out.xdmf")
file_results.parameters["flush_output"] = True
file_results.parameters["functions_share_mesh"] = True
file_results.write(u_out, 0.)
file_results.write(sigma_u_out, 0.)


file_results = XDMFFile("box_in.xdmf")
file_results.parameters["flush_output"] = True
file_results.parameters["functions_share_mesh"] = True
file_results.write(u_in, 0.)
file_results.write(sigma_u_in, 0.)