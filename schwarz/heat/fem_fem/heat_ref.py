from dolfin import *
# from vtkplotter.dolfin import plot
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Create mesh and define function space
    nx = ny = 16
    mesh = UnitSquareMesh(nx, ny)
    V = FunctionSpace(mesh, 'P', 1)

    plot(mesh)
    plt.savefig("square.png")
    plt.close()

    T = 2.0
    num_steps = 10
    dt = T/num_steps
    alpha = 3
    beta = 1.2
    u_D = Expression('1+x[0]*x[0]+alpha*x[1]*x[1]+beta*t', degree=2, alpha=alpha, beta=beta, t=0)

    def boundary(x, on_boundary):
        return on_boundary

    bc = DirichletBC(V, u_D, boundary)

    u_n = project(u_D, V)
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