"""
Minglang Yin, minglang_yin@brown.edu
Multiscale Modeling: FEM + FEM
Zanolli patching method
"""
import numpy as np
from utils import *
from solver import *
from coupling import *

def main():
    ########## Mesh ##########
    mesh1 = IntervalMesh(51, 0, 0.5)
    x1 = np.linspace(0, 0.5, 51)[:, None]

    mesh2 = IntervalMesh(51, 0.5, 1.0)
    x2 = np.linspace(0.5, 1.0, 51)[:, None]

    x_intf = 0.5
    u_true = Expression('C1*exp(x[0]) - C1*exp(-x[0])', degree=2, C1 = np.e/(np.e**2+1) )
    u_o = Expression('0', degree=1)
    
    ########## FEM 1 ##########
    V1 = FunctionSpace(mesh1, "P", 5)
    u1 = TrialFunction(V1)
    v1 = TestFunction(V1)
    F1 = -dot(grad(u1), grad(v1))*dx - u1*v1*dx
    a1, L1 = lhs(F1), rhs(F1)
    u1 = Function(V1)

    model_fem1 = solver_fem(a1, L1, u1)
    u_e1 = interpolate(u_true, V1)

    bc1_l = DirichletBC(V1, u_true, "x[0] < DOLFIN_EPS")

    ########## FEM 2 ##########
    V2 = FunctionSpace(mesh2, "P", 5)
    u_e2 = interpolate(u_true, V2)

    def right_boundary ( x, on_boundary ):
        return on_boundary and ( abs ( x[0] - 1.0 ) < DOLFIN_EPS )
    u_right  = u_e2( 1.0 )
    bc2_r = DirichletBC ( V2, u_right, right_boundary )

    ## multiscale coupling
    eps = 1e-10
    niter = 50
    lmd = 0.0
    theta = 0.25
    err_his = []
    dyn_theta = 1
    for iter in range(0, niter):
        ## model 1
        g1 = Expression("alpha", degree = 1, alpha = lmd)
        bc1_r = DirichletBC(V1, g1, "abs(x[0]-0.5) < DOLFIN_EPS")
        u1 = model_fem1.forward([bc1_l, bc1_r])
        u1_x = project(u1.dx(0), V1)
        u_intf_x = u1_x(x_intf)

        ## model 2
        u2 = TrialFunction(V2)
        v2 = TestFunction(V2)
        g2 = Expression("alpha", degree = 1, alpha = u_intf_x)
        F2_tmp = -dot(grad(u2), grad(v2))*dx - u2*v2*dx - g2*v2*ds
        a2, L2 = lhs(F2_tmp), rhs(F2_tmp)
        u2 = Function(V2)
        solve(a2 == L2, u2, bc2_r)

        ## dynamic theta
        if dyn_theta == 1:
            if iter == 0:
                u1_old = u1(x_intf)
                u2_old = u2(x_intf)
            else:
                u1_new = u1(x_intf)
                u2_new = u2(x_intf)
                e1 = u1_new - u1_old
                e2 = u2_new - u2_old
                theta = (e1 * (e1 - e2)).sum()/((e1 - e2)**2).sum()

        ## lambda
        lmd = theta*u2(x_intf) + (1 - theta)*lmd

        ## error 
        err_1 = errornorm(u_e1, u1, 'L2')
        err_2 = errornorm(u_e2, u2, 'L2')
        err_his.append( np.array([err_1, err_2]) )
        print(f'iter: {iter}, err_1: {err_1:3g}, err_2: {err_2:3g}')

        # if err_1 <= eps and err_2 <= eps:
        #     break
    
    err_his = np.array(err_his)
    
    ##### figures 1: u_s #####
    # fig = plt.figure(constrained_layout=False, figsize=(5, 5))
    # gs = fig.add_gridspec(1, 1)
    # ax = fig.add_subplot(gs[0])

    # h = plot(u1)
    # plt.savefig("u1.png")
    # plt.close()

    ##### figures 4: convergence #####
    fig = plt.figure(constrained_layout=False, figsize=(4, 4))
    gs = fig.add_gridspec(1, 1)
    ax = fig.add_subplot(gs[0])
    ax.plot(err_his[:, 0], color='blue', label='$u1$ (FEM1)')
    ax.plot(err_his[:, 1], color='red', label='$u2$ (FEM2)')
    # ax.axhline(y=eps, linestyle='--', color='black',  label='$\epsilon$')

    ax.set_xlabel('# iter')
    ax.set_ylabel('$L_{2}$ error')
    ax.set_yscale('log')
    ax.legend()
    plt.savefig("convergence.png")
    plt.close()

    np.savetxt("err.dat", err_his)

if __name__ == "__main__":
    main()