from fenics import *
import numpy as np
import matplotlib.pyplot as plt

def main():

    # Create mesh and define function space
    nx = ny = 8
    mesh = UnitSquareMesh(nx, ny)
    V = FunctionSpace(mesh, 'P', 1)

    plot(mesh)
    plt.savefig("square.png")

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
    for n in range(num_steps):
        t += dt
        u_D.t = t

        solve(a==L, u, bc)
        plot(u)
        u_e = interpolate(u_D, V)
        error = np.abs(np.array(u_e.vector()) - np.array(u.vector())).max()
        print('t = %.2f: error = %.3g' %(t, error))
        u_n.assign(u)

if __name__ == "__main__":
    main()