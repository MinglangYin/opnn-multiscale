from fenics import *
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Create mesh and define function space
    nx = ny = 32
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
    
    x = V.tabulate_dof_coordinates()
    x = x.reshape((-1, mesh.geometry().dim()))
    t_vec = np.ones((x.shape[0], 1))*t
    u_tmp = np.array(u_n.vector())[:, None]
    txyu = []
    txyu.append(np.concatenate((t_vec, x, u_tmp), axis=1))

    for n in range(num_steps):
        t += dt
        u_D.t = t

        solve(a==L, u, bc)
        
        ## save
        h = plot(u)
        name = "./fig/"+str(n)+".png"
        plt.colorbar(h)
        plt.savefig(name)
        plt.close()

        ## error
        u_e = interpolate(u_D, V)
        error = np.abs(np.array(u_e.vector()) - np.array(u.vector())).max()
        print('t = %.2f: error = %.3g' %(t, error))
        u_n.assign(u)

        u_tmp = np.array(u.vector())[:, None]
        t_vec = np.ones((x.shape[0], 1))*t
        txyu_tmp = np.concatenate((t_vec, x, u_tmp), axis=1)
        txyu.append(txyu_tmp)

    for id, txyu_tmp in enumerate(txyu):
        name = "data/txyu_" + str(id) + ".dat"
        np.savetxt(name, txyu_tmp, fmt='%1.6f')

if __name__ == "__main__":
    main()