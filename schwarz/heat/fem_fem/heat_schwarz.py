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

def get_true_u(x, t):
    return 1 + x**2 + alpha*y**2 + beta*t

def main():
    # Create mesh and define function space
    niter = 20
    T = 2.0
    num_steps = 10
    dt = T/num_steps
    alpha = 3
    beta = 1.2

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

    u_D = Expression('1+x[0]*x[0]+alpha*x[1]*x[1]+beta*t', degree=2, alpha=alpha, beta=beta, t=0)

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


    u_es = project(u_D, mesh_s)
    u_ep = project(u_D, mesh_p)

    u = TrialFunction(V)
    v = TestFunction(V)
    f = Constant(beta - 2 - 2*alpha)

    F = u*v*dx + dt*dot(grad(u), grad(v))*dx - (u_n + dt*f)*v*dx
    a, L = lhs(F), rhs(F)

    ## Time stepping
    u = Function(V)
    t = 0
    prefix = './figs/u_'
    for n in range(num_steps):
        t += dt
        u_D.t = t

        solve(a==L, u, bc)
        u_e = interpolate(u_D, V)
        error = np.abs(np.array(u_e.vector()) - np.array(u.vector())).max()
        print('t = %.2f: error = %.3g' %(t, error))
        u_n.assign(u)

        ## figures
        name = prefix + str(n) + '.png'
        
        u_range = np.linspace(0, 5, 3)[:,None]

        fig = plt.figure(constrained_layout=False, figsize=(11, 3))

        gs = fig.add_gridspec(1, 3)
        ax = fig.add_subplot(gs[0])
        h = plot(u, cmap='coolwarm', vmin=0, vmax=7)
        ax.set_title('prediction')  
        plt.colorbar(h, ax=ax)

        ax = fig.add_subplot(gs[1])
        h = plot(u_e, cmap='coolwarm', vmin=0, vmax=7)
        ax.set_title('true')
        plt.colorbar(h, ax=ax)

        ax = fig.add_subplot(gs[2])
        h = plot(abs(u-u_e), cmap='coolwarm', vmin=0, vmax=0.0001)
        ax.set_title('err')
        plt.colorbar(h, ax=ax)

        plt.savefig(name)
        plt.close()


if __name__ == "__main__":
    main()