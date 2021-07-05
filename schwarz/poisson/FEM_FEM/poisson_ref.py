"""
Minglang Yin, minglang_yin@brown.edu
Multiscale Modeling: FEM + FEM
In/Out domain, reference solution, no iteration
"""

import numpy as np
import matplotlib.pyplot as plt

from dolfin import *
from mshr import *

def main():
    ## parameter
    niter = 100

    ## define the non-conforming mesh
    box_out = Rectangle(Point(0,0), Point(1.0, 1.0))
    box_in = Rectangle(Point(0.25, 0.25), Point(0.75, 0.75))
    box_hollow = box_out - box_in
    
    mesh1 = generate_mesh(box_hollow, 16)
    mesh2 = generate_mesh(box_in, 16)

    # plot(mesh1)
    # plt.savefig("mesh.png")
    # plt.close()

    ## function space
    V1 = FunctionSpace(mesh1, 'P', 1) ## outer mesh
    V2 = FunctionSpace(mesh2, 'P', 1) ## inner mesh

    ## define boundary condition
    u_true = Expression('1 + x[0]*x[0] + 2*x[1]*x[1]', degree=2)
    u_o = Expression('0', degree=1)

    outer_sq = 'on_boundary && x[0]>=0-DOLFIN_EPS && x[0]<=1+DOLFIN_EPS && x[1]>=0-DOLFIN_EPS && x[1]<=1+DOLFIN_EPS'
    inner_sq = 'on_boundary && x[0]>=0.25-DOLFIN_EPS && x[0]<=0.75+DOLFIN_EPS && x[1]>=0.25-DOLFIN_EPS && x[1]<=0.75+DOLFIN_EPS'
    mesh1_bc_out = DirichletBC(V1, u_true, outer_sq)
    mesh1_bc_in = DirichletBC(V1, u_o, inner_sq)
    mesh2_bc = DirichletBC(V2, u_o, inner_sq)

    ## true solution
    u_e1 = interpolate(u_true, V1)
    u_e2 = interpolate(u_true, V2)

    err_his = np.zeros((niter, 2)) ## [err_in, err_out]

    ## Outside(mesh1), define the variational form
    u1 = TrialFunction(V1)
    v1 = TestFunction(V1)
    f1 = Constant(-6.0)
    a1 = dot(grad(u1), grad(v1))*dx
    L1 = f1*v1*dx
    u1 = Function(V1)

    ## Inside(mesh2): define the variational form
    u2 = TrialFunction(V2)
    v2 = TestFunction(V2)
    f2 = Constant(-6.0)
    a2 = dot(grad(u2), grad(v2))*dx
    L2 = f2*v2*dx
    u2 = Function(V2)

    ## Schwarz alternating method
    for iter in range(0, 1):
        if iter != 0:
            ## boundary condition
            mesh1_bc_in = DirichletBC(V1, u_o, inner_sq)
            mesh2_bc = DirichletBC(V_in, u_true, boundary)

        ## solve
        solve(a1 == L1, u1, [mesh1_bc_out, mesh1_bc_in])
        solve(a2 == L2, u2, mesh2_bc)

        ## error 
        error_out = np.abs(np.array(u_e1.vector()) - np.array(u1.vector())).max()
        error_in = np.abs(np.array(u_e2.vector()) - np.array(u2.vector())).max()
        err_his[iter, :] = np.array((error_out, error_in))
        print(f'iter: {iter}, error_out: {error_out:3g}, error_in: {error_in:3g}')
        

    ## Plot
    plot(u1, vmin=0, vmax=4)
    plot(u2, vmin=0, vmax=4)
    
    plt.savefig("u.png")
    plt.close()


if __name__ == "__main__":
    main()