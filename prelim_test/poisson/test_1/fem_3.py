"""
Author: Minglang Yin. minglang_yin@brown.edu

Coupling a FEM solver with a DeepONet for poisson equation

div(grad(u)) = -f, Omage = [0, 1]*[0, 1], f = 1
"""

import numpy as np
import matplotlib.pyplot as plt

from fenics import *
from scipy.interpolate import interp2d

# u_interp = interpolate(u_int, V) ## interpolate an expression onto a mesh

def main():
    ## create mesh and define function space
    num_cases = 100 # 100 
    mesh_size = 32
    mesh = UnitSquareMesh(mesh_size, mesh_size)
    V = FunctionSpace(mesh, 'P', 1)

    ## define the variational form
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Constant(-6.0)
    a = dot(grad(u), grad(v))*dx
    L = f*v*dx

    u_D = Expression('a + b*x[0]*x[0] + c*x[1]*x[1]', a=1, b=1, c=2, degree=2)

    ## boundary condition
    def boundary(x, on_boundary):
        return on_boundary
    bc = DirichletBC(V, u_D, boundary)

    ## solve
    u = Function(V)
    solve(a == L, u, bc)

    ## save txt
    x = V.tabulate_dof_coordinates()
    x = x.reshape((-1, mesh.geometry().dim()))
    u_dump = np.array(u.vector())[:, None]
    name = "data_single/xyu.dat"
    np.savetxt(name, np.concatenate((x[:,0:1], x[:,1:2], u_dump), axis=1) )

    ## plot 
    plot(u)
    name = 'fig/u.png'
    plt.savefig(name)
    plt.close()


if __name__ == "__main__":
    main()