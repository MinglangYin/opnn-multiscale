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
force = 1
dim = 11
eps = 1e-10
niter = 10
theta = 0.25

## mesh
square_out = Rectangle(Point(0,0), Point(1.0, 1.0))
square_in = Rectangle(Point(0.3,0.0), Point(0.7, 0.7))
mesh_out = generate_mesh(square_out - square_in, 32)
mesh_in = generate_mesh(square_in, 32)

x_out = mesh_out.coordinates()
x_in = mesh_in.coordinates()

x_v = np.linspace(0.0, 0.7, dim)[:, None]
x_h = np.linspace(0.3, 0.7, dim)[:, None]
ones = np.ones((dim, 1))
x_intf_top = np.concatenate((x_h, ones*0.7), axis=1)
x_intf_left = np.concatenate((ones*0.3, x_v), axis=1)
x_intf_right = np.concatenate((ones*0.7, x_v), axis=1)
x_intf_bot = np.concatenate((x_h, ones*0.0), axis=1)
x_intf = np.concatenate((x_intf_top, x_intf_left, x_intf_right), axis=0)

x_lin = np.linspace(0.0, 1.0, dim)[:, None]
ones = np.ones((dim, 1))
x_out_top = np.concatenate((x_lin, ones), axis=1)
x_out_left = np.concatenate((ones*0.0, x_lin), axis=1)
x_out_right = np.concatenate((ones, x_lin), axis=1)
x_out_b = np.concatenate((x_out_top, x_out_left, x_out_right), axis=0)

## model out
W_out = TensorFunctionSpace(mesh_out, "P", 2)
V_out = VectorFunctionSpace(mesh_out, "P", 2)
def clamped_boundary(x, on_boundary):
    return on_boundary and x[1] < eps
bc_out_bot = DirichletBC(V_out, Constant((0, 0)), clamped_boundary)

boundaries = MeshFunction("size_t", mesh_out, mesh_out.topology().dim() - 1)
boundaries.set_all(0)

top = AutoSubDomain(lambda x: near(x[1], 1))
top.mark(boundaries, 1)
ds_out = ds(subdomain_data=boundaries)

def in_boundary(x, on_boundary):
    return on_boundary and x[0] >= 0.3-DOLFIN_EPS and x[0] <= 0.7+DOLFIN_EPS \
        and x[1] <= 0.7+DOLFIN_EPS
bc_out_inner = DirichletBC(V_out, ("0", "0"), in_boundary)

u_out = TrialFunction(V_out)
d = u_out.geometric_dimension()
v_out = TestFunction(V_out)
f_out = Constant((0, 0))
T_out = Constant((0, force)) ## traction on the top surface
a_out = inner(sigma(u_out), epsilon(v_out))*dx
L_out = dot(f_out, v_out)*dx + dot(T_out, v_out)*ds_out(1) ## ds is on the boundary

##### The above is clean #####
##############################
##############################


## model in
W_in = TensorFunctionSpace(mesh_in, "P", 2)
V_in = VectorFunctionSpace(mesh_in, "P", 2)

boundaries_in = MeshFunction("size_t", mesh_in, mesh_in.topology().dim() - 1)
boundaries_in.set_all(0)

top = AutoSubDomain(lambda x: near(x[1], 0.7))
top.mark(boundaries_in, 1)
left = AutoSubDomain(lambda x: near(x[0], 0.3))
left.mark(boundaries_in, 2)
right = AutoSubDomain(lambda x: near(x[0], 0.7))
right.mark(boundaries_in, 3)
ds_in = ds(subdomain_data=boundaries_in)

bc_in_bot = DirichletBC(V_in, Constant((0, 0)), clamped_boundary)

## true solution
u_e = get_true_sol()
u_e_out = projection(u_e, x_out)
u_e_in = projection(u_e, x_in)

## inner boundary displacement
# def inner_boundary(x, on_boundary):
#     return on_boundary and abs(x[0] - 0.3)<tol or abs(x[0] - 0.7)<tol or abs(x[1] - 0.3)<tol or abs(x[1] - 0.7)<tol

dir_top = np.array([[0, 0], [-1, 0], [0, 0], [0, -1]])
# dir_bot = np.array([[0, 0], [-1, 0], [0, 0], [0, -1]])
dir_left = np.array([[1, 0], [0, 0], [0, 1], [0, 0]])
dir_right = np.array([[-1, 0], [0, 0], [0, -1], [0, 0]])

err_his = []
dyn_theta = 0
for iter in range(0, niter):
    if iter == 0:
        ## model(out)
        u_out = Function(V_out)
        solve(a_out == L_out, u_out, [bc_out_bot, bc_out_inner])

    else:
        ## model(out)
        u_out_b = projection(u_out, x_out_b)
        ux1_tmp = np.concatenate((x_out_b, u_out_b), axis=1)
        ux2_tmp = np.concatenate((x_intf, lmd), axis=1)
        ux_tmp = np.concatenate((ux1_tmp, ux2_tmp), axis=0)

        bc_out_exp = data2exp_2d(ux_tmp, V_out) ## careful with the interpolation

        ######
        V_out = VectorFunctionSpace(mesh_out, "P", 2)
        bc_out_bot = DirichletBC(V_out, Constant((0, 0)), clamped_boundary)

        bc_out_inner = DirichletBC(V_out, bc_out_exp, in_boundary)

        u_out = TrialFunction(V_out)
        v_out = TestFunction(V_out)
        f_out = Constant((0, 0))
        T_out = Constant((0, force))
        a_out = inner(sigma(u_out), epsilon(v_out))*dx
        L_out = dot(f_out, v_out)*dx + dot(T_out, v_out)*ds_out(1) ## ds is on the boundary

        u_out = Function(V_out)
        solve(a_out == L_out, u_out, [bc_out_bot, bc_out_inner])

    ## model(in)
    # W_in = TensorFunctionSpace(mesh_in, "P", 2)
    sigma_out = project(sigma(u_out), W_out)
    sigma_intf_top = -np.matmul(projection(sigma_out, x_intf_top), dir_top)
    sigma_intf_left = -np.matmul(projection(sigma_out, x_intf_left), dir_left)
    sigma_intf_right = -np.matmul(projection(sigma_out, x_intf_right), dir_right)

    ux_top = np.concatenate((x_intf_top, sigma_intf_top), axis=1)
    ux_left = np.concatenate((x_intf_left, sigma_intf_left), axis=1)
    ux_right = np.concatenate((x_intf_right, sigma_intf_right), axis=1)

    # print(ux_top)
    # print(ux_left)
    # print(ux_right)
    # exit()

    u_intf_bot = np.zeros_like(x_intf_bot)
    ux_bot = np.concatenate((x_intf_bot, u_intf_bot), axis=1)

    ux_t = np.concatenate((ux_top, ux_bot), axis=0)
    ux_lr = np.concatenate((ux_left, ux_right), axis=0)
    f_t = data2exp_2d(ux_t, V_in)
    f_lr = data2exp_2d(ux_lr, V_in)

    f_in = Constant((0, 0))
    u_in = TrialFunction(V_in)
    v_in = TestFunction(V_in)
    a_in = inner(sigma(u_in), epsilon(v_in))*dx
    L_in = dot(f_in, v_in)*dx + dot(f_t, v_in)*ds_in(1) + dot(f_lr, v_in)*ds_in(2) + dot(f_lr, v_in)*ds_in(3)

    u_in = Function(V_in)
    solve(a_in == L_in, u_in, bc_in_bot)


    ## stress u_in
    sigma_in = project(sigma(u_in), W_in)
    sigma_in_top = -np.matmul(projection(sigma_in, x_intf_top), dir_top)
    sigma_in_left = -np.matmul(projection(sigma_in, x_intf_left), dir_left)
    sigma_in_right = -np.matmul(projection(sigma_in, x_intf_right), dir_right)

    print(sigma_in_top, sigma_intf_top)
    print(sigma_in_left, sigma_intf_left)
    print(sigma_in_right, sigma_intf_right)
    exit()


    # file_results = XDMFFile("box_out.xdmf")
    # file_results.parameters["flush_output"] = True
    # file_results.parameters["functions_share_mesh"] = True
    # file_results.write(u_out, 0.)

    # file_results = XDMFFile("box_in.xdmf")
    # file_results.parameters["flush_output"] = True
    # file_results.parameters["functions_share_mesh"] = True
    # file_results.write(u_in, 0.)



    exit()

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