"""
Minglang Yin, minglang_yin@brown.edu
u_xx - u = 0, u(0) = 0, u_x(1) = 1
"""
import numpy as np
import matplotlib.pyplot as plt

from fenics import *

def main():
    mesh = IntervalMesh(101,0,1.0)

    u_true = Expression('C1*exp(x[0]) - C1*exp(-x[0])', degree=2, C1 = np.e/(np.e**2+1) )
    u_o = Expression('0', degree=1)
    
    V = FunctionSpace(mesh, "P", 1)
    u = TrialFunction(V)
    v = TestFunction(V)

    F = -dot(grad(u), grad(v))*dx - u*v*dx
    a, L = lhs(F), rhs(F)
    
    tol = 1E-14
    def boundary(x, on_boundary):
        return on_boundary
    bc = DirichletBC(V, u_true, boundary)

    u = Function(V)
    solve(a == L, u, bc)

    u_e = interpolate(u_true, V)

    plot(u - u_e)
    # plot(u_e)
    plt.savefig('u.png')

if __name__ == "__main__":
    main()