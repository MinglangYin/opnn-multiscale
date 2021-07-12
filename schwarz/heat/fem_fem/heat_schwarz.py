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
    interpolant = interp2d(x, y, values, kind='cubic', copy=False, bounds_error=True)

    class ExpressionFromScipyFunction(UserExpression):
        def __init__(self, f, **kwargs):
            self._f = f
            UserExpression.__init__(self, **kwargs)
        def eval(self, values, x):
            values[:] = self._f(*x)

    u = ExpressionFromScipyFunction(interpolant, element=V.ufl_element())
    return u

def get_true_u(x, t, alpha, beta):
    return 1 + x[:, 0]**2 + alpha*x[:, 1]**2 + beta*t

def main():
    # Create mesh and define function space
    niter = 20
    T = 2.0
    num_steps = 10
    dt = T/num_steps
    alpha = 3
    beta = 1.2
    eps = 0.001

    gemma1_x, gemma1_y = 0.4, 0.6
    gemma2_x, gemma2_y = 0.2, 0.8

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

    u_true = Expression('1+x[0]*x[0]+alpha*x[1]*x[1]+beta*t', degree=2, alpha=alpha, beta=beta, t=0)
    u_o = Expression('0', degree=1)

    s_go = 'on_boundary && x[0]>=0-DOLFIN_EPS && x[0]<=1+DOLFIN_EPS &&\
            x[1]>=0-DOLFIN_EPS && x[1]<=1+DOLFIN_EPS' # mesh_s outer boundary, need to be consistent
    s_g1 = 'on_boundary && x[0]>=0.4-DOLFIN_EPS && x[0]<=0.6+DOLFIN_EPS &&\
            x[1]>=0.4-DOLFIN_EPS && x[1]<=0.6+DOLFIN_EPS' # mesh_s on Gemma 1
    p_g2 = 'on_boundary'# && abs(x[0]-0.3)<=DOLFIN_EPS && abs(x[0]-0.7)<=DOLFIN_EPS &&\
            #abs(x[1]-0.3)<=DOLFIN_EPS && abs(x[1]-0.7)<=DOLFIN_EPS' # mesh_p on Gemma 2
    mesh_s_go = DirichletBC(V_s, u_true, s_go)
    mesh_s_g1 = DirichletBC(V_s, u_o, s_g1)
    mesh_p_g2 = DirichletBC(V_p, u_o, p_g2)

    ## true solution
    u_es = project(u_true, V_s)
    u_ep = project(u_true, V_p)

    ## boundary coordinates
    dim = 10
    top_o, bot_o, left_o, right_o = get_bound_coord([0, 0], [1.0, 1.0], dim)
    top_g1, bot_g1, left_g1, right_g1 = get_bound_coord([gemma1_x, gemma1_x], [gemma1_y, gemma1_y], dim)
    top_g2, bot_g2, left_g2, right_g2 = get_bound_coord([gemma2_x, gemma2_x], [gemma2_y, gemma2_y], dim)

    g1_coord = np.concatenate((top_g1, bot_g1, left_g1, right_g1), axis=0)
    g2_coord = np.concatenate((top_g2, bot_g2, left_g2, right_g2), axis=0)
    o_coord = np.concatenate((top_o, bot_o, left_o, right_o), axis=0)
    u_s_out = get_true_u(o_coord, 0, alpha, beta)[:, None]

    ## mesh_s, variational form
    u_s_n = project(u_true, V_s)
    u_s = TrialFunction(V_s)
    v_s = TestFunction(V_s)
    f_s = Constant(beta - 2 - 2*alpha)

    F_s = u_s*v_s*dx + dt*dot(grad(u_s), grad(v_s))*dx - (u_s_n + dt*f_s)*v_s*dx
    a_s, L_s = lhs(F_s), rhs(F_s)

    ## mesh_p, variational form
    u_p_n = project(u_true, V_p)
    u_p = TrialFunction(V_p)
    v_p = TestFunction(V_p)
    f_p = Constant(beta - 2 - 2*alpha)

    F_p = u_p*v_p*dx + dt*dot(grad(u_p), grad(v_p))*dx - (u_p_n + dt*f_p)*v_p*dx
    a_p, L_p = lhs(F_p), rhs(F_p)

    ## Schwarz alternating method
    tau_n1 = 0.5
    # u_s_g1_old = np.zeros((dim*4, 1))
    u_s_g1_new = np.zeros((dim*4, 1)) #
    u_s_g2_old = np.zeros((dim*4, 1)) #
    u_s_g2_new = np.zeros((dim*4, 1)) #

    u_p_g1_old = np.zeros((dim*4, 1)) #
    # u_p_g1_new = np.zeros((dim*4, 1))
    u_p_g2_old = np.zeros((dim*4, 1)) #
    u_p_g2_new = np.zeros((dim*4, 1)) #

    ## Time stepping
    u_s = Function(V_s)
    u_p = Function(V_p)
    err_his = [] ## [err_in, err_out]

    t = 0
    prefix = './figs/u_'
    for n in range(num_steps):
        t += dt
        u_true.t = t
        err_his_tmp = []

        ## schwarz alternating method
        for iter in range(0, niter):
            if iter == 0:
                solve(a_s == L_s, u_s, [mesh_s_go, mesh_s_g1])
                solve(a_p == L_p, u_p, mesh_p_g2)

                for id, coord in enumerate(g2_coord):
                    u_s_g2_old[id] = u_s(coord)
                    u_p_g2_old[id] = u_s(coord)

            else:
                ## step a
                for id, coord in enumerate(g1_coord):
                    u_p_g1_old[id] = u_p(coord)
                u_s_g1_new = np.copy(u_p_g1_old)

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
                u_p_bc_exp = data2exp(np.concatenate((g2_coord, u_p_g2_new), axis=1), V_p)
                mesh_p_g2 = DirichletBC(V_p, u_p_bc_exp, p_g2)
                solve(a_p == L_p, u_p, mesh_p_g2)

                ## update
                u_p_g2_old = np.copy(u_p_g2_new)
                tau_n1 = np.copy(tau_n)

            ## error 
            u_ep = interpolate(u_true, V_p)
            u_es = interpolate(u_true, V_s)
            error_s = ((np.array(u_es.vector()) - np.array(u_s.vector()))**2).mean()
            error_p = ((np.array(u_ep.vector()) - np.array(u_p.vector()))**2).mean()
            err_his_tmp.append( np.array([error_s, error_p]) )
            print(f'iter: {iter}, error_s: {error_s:3g}, error_p: {error_p:3g}')
            u_p_n.assign(u_p)

            # if error_p <= eps and error_s <= eps:
            #     break

        err_his.append(err_his_tmp)
        name = prefix + str(n) + '.png'
        h = plot(abs(u_p - u_ep), cmap='coolwarm')
        plt.colorbar(h)
        plt.savefig(name)
        plt.close()

    
    fig = plt.figure(constrained_layout=False, figsize=(5, 6))
    gs = fig.add_gridspec(1, 1)
    ax = fig.add_subplot(gs[0])
    for err_his_tmp in err_his:
        h = ax.plot(err_his_tmp)
    ax.set_yscale('log')
    plt.savefig('convergence.png')
    plt.close()



    ## figures
    # name = prefix + str(n) + '.png'
    # fig = plt.figure(constrained_layout=False, figsize=(11, 3))

    # gs = fig.add_gridspec(1, 3)
    # ax = fig.add_subplot(gs[0])
    # h = plot(u, cmap='coolwarm', vmin=0, vmax=7)
    # ax.set_title('prediction')  
    # plt.colorbar(h, ax=ax)

    # ax = fig.add_subplot(gs[1])
    # h = plot(u_e, cmap='coolwarm', vmin=0, vmax=7)
    # ax.set_title('true')
    # plt.colorbar(h, ax=ax)

    # ax = fig.add_subplot(gs[2])
    # h = plot(abs(u-u_e), cmap='coolwarm', vmin=0, vmax=0.0001)
    # ax.set_title('err')
    # plt.colorbar(h, ax=ax)

    # plt.savefig(name)
    # plt.close()


if __name__ == "__main__":
    main()