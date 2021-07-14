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
    V1 = FunctionSpace(mesh1, "P", 1)
    u1 = TrialFunction(V1)
    v1 = TestFunction(V1)
    F1 = -dot(grad(u1), grad(v1))*dx - u1*v1*dx
    a1, L1 = lhs(F1), rhs(F1)
    u1 = Function(V1)

    model_fem1 = solver_fem(a1, L1, u1)
    u_e1 = interpolate(u_true, V1)

    bc1_l = DirichletBC(V1, u_true, "x[0] < DOLFIN_EPS")
    bc1_r = DirichletBC(V1, u_o, "x[0]-0.5 < DOLFIN_EPS")
    
    ########## FEM 2 ##########
    V2 = FunctionSpace(mesh2, "P", 1)
    u2 = TrialFunction(V2)
    v2 = TestFunction(V2)
    F2 = -dot(grad(u2), grad(v2))*dx - u2*v2*dx
    a2, L2 = lhs(F2), rhs(F2)
    # u2 = Function(V2)
    
    # model_fem2 = solver_fem(a2, L2, u2)
    u_e2 = interpolate(u_true, V2)

    # bc2_l = DirichletBC(V2, u_o, "x[0]-0.5 < DOLFIN_EPS")
    def right_boundary ( x, on_boundary ):
        return on_boundary and ( abs ( x[0] - 1.0 ) < DOLFIN_EPS )
    u_right  = u_e2( 1.0 )
    bc = DirichletBC ( V2, u_right, right_boundary )

    


    # mesh = UnitIntervalMesh ( 10 )
    # ue = Expression('C1*exp(x[0]) - C1*exp(-x[0])', degree=2, C1 = np.e/(np.e**2+1) )
    # g = Expression ( "1.0", degree = 10 )
    # V = FunctionSpace ( mesh, "CG", 1 )
    # def right_boundary ( x, on_boundary ):
    #     return on_boundary and ( abs ( x[0] - 1.0 ) < DOLFIN_EPS )
    # u_right  = ue ( 1.0 )
    # bc = DirichletBC ( V, u_right, right_boundary )
    # u = TrialFunction ( V )
    # v = TestFunction ( V )
    # A   = -inner ( grad ( u ), grad ( v ) ) * dx - u * v * dx 
    # RHS = g * v * ds
    # u = Function ( V )
    # solve ( A == RHS, u, bc )

    ## multiscale coupling
    eps = 0.0001
    niter = 20

    for iter in range(0, niter):
        u_1 = model_fem1.forward([bc1_l, bc1_r])
        u_1_x = project(u_1.dx(0), V1)
        u_intf_x = u_1_x(x_intf)

        g = Expression("alpha", degree = 1, alpha = u_intf_x)

        F2_tmp = -dot(grad(u2), grad(v2))*dx - u2*v2*dx - 1.0*v2*ds
        a2, L2 = lhs(F2_tmp), rhs(F2_tmp)
        u2 = Function(V2)
        solve(a2 == L2, u2, bc)
        exit()


        ## error 
        # if norm == "L2":
        #     error_s = l2_norm(.u_es, .u_s)
        #     error_p = l2_norm(.u_ep, .u_p)
        # err_his.append( np.array([error_s, error_p]) )
        # print(f'iter: {iter}, error_s: {error_s:3g}, error_p: {error_p:3g}')

        # if error_p <= eps and error_s <= eps:
        #     break
    
    err_his = np.array(err_his)
    
    ##### figures 1: u_s #####
    fig = plt.figure(constrained_layout=False, figsize=(11, 3))
    gs = fig.add_gridspec(1, 3)
    ax = fig.add_subplot(gs[0])
    h = plot(u_s)
    plt.colorbar(h)
    ax.set_title('prediction')

    ax = fig.add_subplot(gs[1])
    h = plot(u_es)
    plt.colorbar(h)
    ax.set_title('true')

    ax = fig.add_subplot(gs[2])
    h = plot(abs(u_s - u_es))
    plt.colorbar(h)
    ax.set_title('err')

    plt.savefig("u_s.png")
    plt.close()

    ##### figures 2: u_p #####
    fig = plt.figure(constrained_layout=False, figsize=(11, 3))
    gs = fig.add_gridspec(1, 3)
    ax = fig.add_subplot(gs[0])
    h = plot(u_p)
    plt.colorbar(h)
    ax.set_title('prediction')

    ax = fig.add_subplot(gs[1])
    h = plot(u_ep)
    plt.colorbar(h)
    ax.set_title('true')

    ax = fig.add_subplot(gs[2])
    h = plot(abs(u_p - u_ep))
    plt.colorbar(h)
    ax.set_title('err')

    plt.savefig("u_p.png")
    plt.close()

    ##### figures 3: mesh #####
    fig = plt.figure(constrained_layout=False, figsize=(11, 3))
    gs = fig.add_gridspec(1, 3)
    ax = fig.add_subplot(gs[0])
    plot(mesh_s, alpha=0.5)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_title("mesh s")

    ax = fig.add_subplot(gs[1])
    plot(mesh_p, alpha=0.5)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_title("mesh p")

    ax = fig.add_subplot(gs[2])
    plot(mesh_p, alpha=0.5)
    plot(mesh_s, alpha=0.5)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_title("mesh s+p")

    plt.savefig("mesh.png")
    plt.close()

    ##### figures 4: convergence #####
    fig = plt.figure(constrained_layout=False, figsize=(4, 4))
    gs = fig.add_gridspec(1, 1)
    ax = fig.add_subplot(gs[0])
    ax.plot(err_his[:, 0], color='blue', label='$u_{s}$ (FEM)')
    ax.plot(err_his[:, 1], color='red', label='$u_{p}$ (DeepONet)')
    ax.axhline(y=eps, linestyle='--', color='black',  label='$\epsilon$')

    ax.set_xlabel('# iter')
    ax.set_ylabel('$L_{2}$ error')
    ax.set_yscale('log')
    ax.legend()
    plt.savefig("convergence.png")
    plt.close()

if __name__ == "__main__":
    main()