"""
Minglang Yin, minglang_yin@brown.edu
Multiscale Modeling: FEM + FEM
Schwarz alternating method
"""
import numpy as np
import matplotlib.pyplot as plt

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
    # interf_coord = np.concatenate((top, bot, left, right), axis=0)

    return top, bot, left, right

def data2exp(data, V):
    ## data
    x, y, values = data[:,0], data[:,1], data[:,2]
    interpolant = interp2d(x, y, values, kind='linear', copy=False, bounds_error=True)

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

def main():
    ## parameter
    niter = 3
    eps = 0.01
    gemma1_x, gemma1_y = 0.4, 0.6
    gemma2_x, gemma2_y = 0.3, 0.7

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

    s_bo = 'on_boundary && x[0]>=0-DOLFIN_EPS && x[0]<=1+DOLFIN_EPS &&\
            x[1]>=0-DOLFIN_EPS && x[1]<=1+DOLFIN_EPS' # mesh_s outer boundary, need to be consistent
    s_b1 = 'on_boundary && x[0]>=0.4-DOLFIN_EPS && x[0]<=0.6+DOLFIN_EPS &&\
            x[1]>=0.4-DOLFIN_EPS && x[1]<=0.6+DOLFIN_EPS' # mesh_s on Gemma 1
    p_b2 = 'on_boundary && x[0]>=0.3-DOLFIN_EPS && x[0]<=0.7+DOLFIN_EPS &&\
            x[1]>=0.3-DOLFIN_EPS && x[1]<=0.7+DOLFIN_EPS' # mesh_p on Gemma 2
    mesh_s_bo = DirichletBC(V_s, u_true, s_bo)
    mesh_s_b1 = DirichletBC(V_s, u_o, s_b1)
    mesh_p_b2 = DirichletBC(V_p, u_o, p_b2)

    ## true solution
    u_es = interpolate(u_true, V_s)
    u_ep = interpolate(u_true, V_p)
    
    ## boundary coordinates
    dim = 10
    top_o, bot_o, left_o, right_o = get_bound_coord([0, 0], [1.0, 1.0], dim)
    top_g1, bot_g1, left_g1, right_g1 = get_bound_coord([gemma1_x, gemma1_x], [gemma1_y, gemma1_y], dim)
    top_g2, bot_g2, left_g2, right_g2 = get_bound_coord([gemma2_x, gemma2_x], [gemma2_y, gemma2_y], dim)

    g1_coord = np.concatenate((top_g1, bot_g1, left_g1, right_g1), axis=0)
    g2_coord = np.concatenate((top_g2, bot_g2, left_g2, right_g2), axis=0)
    o_coord = np.concatenate((top_o, bot_o, left_o, right_o), axis=0)
    u_s_out = get_true_u(o_coord)[:, None]
    
    ## mesh_s, define the variational form
    u_s = TrialFunction(V_s)
    v_s = TestFunction(V_s)
    f_s = Constant(-6.0)
    a_s = dot(grad(u_s), grad(v_s))*dx
    L_s = f_s*v_s*dx
    u_s = Function(V_s)

    ## mesh_p: define the variational form
    u_p = TrialFunction(V_p)
    v_p = TestFunction(V_p)
    f_p = Constant(-6.0)
    a_p = dot(grad(u_p), grad(v_p))*dx
    L_p = f_p*v_p*dx
    u_p = Function(V_p)

    ## Schwarz alternating method
    tau = 1
    u_s_g1_old = np.zeros((dim*4, 1))
    u_s_g1_new = np.zeros((dim*4, 1))
    u_s_g2_old = np.zeros((dim*4, 1))
    u_s_g2_new = np.zeros((dim*4, 1))

    u_p_g1_old = np.zeros((dim*4, 1))
    u_p_g1_new = np.zeros((dim*4, 1))
    u_p_g2_old = np.zeros((dim*4, 1))
    u_p_g2_new = np.zeros((dim*4, 1))

    err_his = np.zeros((niter, 2)) ## [err_in, err_out]
    for iter in range(0, niter):
        if iter != 0:
            ## step a
            for id, coord in enumerate(g1_coord):
                u_p_g2_old[id] = u_p(coord)
            u_s_g1_new = np.copy(u_p_g2_old)

            u_s_intp = np.concatenate((u_s_g1_new, u_s_out), axis=0)
            x_s_intp = np.concatenate((g1_coord, o_coord), axis=0)
            u_s_bc_exp = data2exp(np.concatenate((x_s_intp, u_s_intp), axis=1), V_s)

            ## step b
            mesh_s_b1 = DirichletBC(V_s, u_s_bc_exp, s_b1)
            solve(a_s == L_s, u_s, [mesh_s_bo, mesh_s_b1])

            ## step c, d
            for id, coord in enumerate(g2_coord):
                u_s_g2_new[id] = u_s(coord)
            # u_s_g1_new = np.copy(u_s_g2_new)

            ## step e (relaxation)
            Q_n = u_s_g2_new - u_p_g2_old
            Q_n1 = u_s_g2_old - u_p_g2_old ## wrong!!
            tau_n = tau + (tau-1)*((Q_n1 - Q_n)*Q_n).sum()/( ((Q_n1 - Q_n)**2).sum() )

            u_p_g2_new = tau*u_p_g2_old + (1-tau)*u_s_g2_new

            ## step f
            u_p_bc_exp = data2exp(np.concatenate((g2_coord, u_p_g2_new), axis=1), V_p)
            mesh_p_b2 = DirichletBC(V_p, u_p_bc_exp, p_b2)
            solve(a_p == L_p, u_p, mesh_p_b2)

            ## step g (useless)

        else:   
            ## solve
            solve(a_s == L_s, u_s, [mesh_s_bo, mesh_s_b1])
            solve(a_p == L_p, u_p, mesh_p_b2)

        ## error 
        error_s = np.abs(np.array(u_es.vector()) - np.array(u_s.vector())).max()
        error_p = np.abs(np.array(u_ep.vector()) - np.array(u_p.vector())).max()
        err_his[iter, :] = np.array((error_s, error_p))
        print(f'iter: {iter}, error_s: {error_s:3g}, error_p: {error_p:3g}')

        if error_p <= eps and error_s <= eps:
            break
        
    ## figures u_s
    fig = plt.figure(constrained_layout=False, figsize=(11, 3))
    gs = fig.add_gridspec(1, 3)
    ax = fig.add_subplot(gs[0])
    plot(u_s, vmin=0, vmax=4)

    ax = fig.add_subplot(gs[1])
    plot(u_es, vmin=0, vmax=4)

    ax = fig.add_subplot(gs[2])
    plot(abs(u_s - u_es), vmin=0, vmax=4)

    plt.savefig("u_s.png")
    plt.close()

    ## figures u_p
    fig = plt.figure(constrained_layout=False, figsize=(11, 3))
    gs = fig.add_gridspec(1, 3)
    ax = fig.add_subplot(gs[0])
    plot(u_p, vmin=0, vmax=4)

    ax = fig.add_subplot(gs[1])
    plot(u_ep, vmin=0, vmax=4)

    ax = fig.add_subplot(gs[2])
    plot(abs(u_p - u_ep), vmin=0, vmax=4)

    plt.savefig("u_p.png")
    plt.close()

    ## mesh 
    fig = plt.figure(constrained_layout=False, figsize=(6, 6))
    gs = fig.add_gridspec(1, 1)
    ax = fig.add_subplot(gs[0])
    plot(mesh_s, alpha=0.5)
    plot(mesh_p, alpha=0.5)
    plt.savefig("mesh.png")
    plt.close()

    ## figure convergence
    fig = plt.figure(constrained_layout=False, figsize=(4, 4))
    gs = fig.add_gridspec(1, 1)
    ax = fig.add_subplot(gs[0])
    ax.plot(err_his[:, 0], color='blue', label='u_{s}')
    ax.plot(err_his[:, 1], color='red', label='u_{p}')
    ax.set_xlabel('# iter')
    ax.set_ylabel('L_{2} error')
    ax.legend()
    plt.savefig("convergence.png")
    plt.close()

if __name__ == "__main__":
    main()