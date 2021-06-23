"""
Author: Minglang Yin. minglang_yin@brown.edu

Coupling a FEM solver with a DeepONet for poisson equation

div(grad(u)) = -f, Omage = [0, 1]*[0, 1], f = 1
"""

import numpy as np
import matplotlib.pyplot as plt

from fenics import *

def main():
    ## create mesh and define function space
    mesh = UnitSquareMesh(8, 8)
    V = FunctionSpace(mesh, 'P', 1)

    ## define boundary condition
    u_D = Expression('1 + x[0]*x[0] + 2*x[1]*x[1]', degree=2)

    def boundary(x, on_boundary):
        return on_boundary

    def boundary_inner(x, on_boundary):
        return 'x[0]>0.25' and 'x[0]<0.75' and 'x[1]>0.25' and 'x[1]<0.75'

    bc = DirichletBC(V, u_D, boundary)

    ## define the variational form
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Constant(-6.0)
    a = dot(grad(u), grad(v))*dx
    L = f*v_dx

    ## solve
    u = Function(V)
    solve(a == L, u, bc)



if __name__ = "__main__":
    main()