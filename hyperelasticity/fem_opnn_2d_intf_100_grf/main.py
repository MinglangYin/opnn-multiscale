"""
## Author: Minglang Yin, minglang_yin@brown.edu
## Uniaxial pulling of a 3D plate
## Holzapfel-Gasser-Odgen Model
## Clamped bottom
## Impose displacement BC on the top
"""
import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

from dolfin import *
from mshr import *
from opnn import *
from utils import *

def main():
    path = 'Results'
    os.makedirs(path, exist_ok=True)

    niter = 20
    eps = 1e-5
    theta = 0.5
    r0 = 0.3

    U_scaling = 10
    S_scaling = 10

    E, nu = 1.0, 0.3
    K1, K2 = 0.1, 1.5
    alpha = np.pi/2      
    kappa = 0

    ## interface and lines
    x_intf = np.loadtxt("x_intf_ext.dat")
    center_coord = [0.5, 0.5]
    N_intf = circ_noraml(x_intf, center_coord) ## assume circle
    dim_intf = x_intf.shape[0]

    ##################
    # id_pick = 21

    # dataset_sph = scipy.io.loadmat("hyper_notch_100_nofrac.mat")
    # top_disp = dataset_sph["disp_top"][id_pick]
    # top_coord_ref = dataset_sph["coord_top_ref"][:, :2]
    # coord_init = dataset_sph["coord_init"]
    # coord = dataset_sph["coord"]
    # disp_sph = coord[id_pick] - coord_init
    # data_true = np.concatenate((coord_init[:, :2], disp_sph[:, :2] ), axis=1) ## coordiante + displacement

    # stress = dataset_sph['snapshot'][id_pick]

    # ## traction
    # intp = 5
    # f_p11 = Rbf(coord_init[::intp, 0], coord_init[::intp, 1], stress[::intp, 0])
    # f_p12 = Rbf(coord_init[::intp, 0], coord_init[::intp, 1], stress[::intp, 1])
    # f_p21 = Rbf(coord_init[::intp, 0], coord_init[::intp, 1], stress[::intp, 3])
    # f_p22 = Rbf(coord_init[::intp, 0], coord_init[::intp, 1], stress[::intp, 4])

    # p11_intf = f_p11(x_intf[:, 0], x_intf[:, 1])
    # p12_intf = f_p12(x_intf[:, 0], x_intf[:, 1])
    # p21_intf = f_p21(x_intf[:, 0], x_intf[:, 1])
    # p22_intf = f_p22(x_intf[:, 0], x_intf[:, 1])

    # Tx_intf_e = p11_intf*N_intf[:, 0] + p12_intf*N_intf[:, 1]
    # Ty_intf_e = p21_intf*N_intf[:, 0] + p22_intf*N_intf[:, 1]

    # Txy_intf_true = np.concatenate((x_intf, Tx_intf_e[:, None], Ty_intf_e[:, None]), axis=1)
    # print( np.concatenate((p11_intf[:, None], p12_intf[:, None], p21_intf[:, None], p22_intf[:, None]), axis = 1) )

    #########################################################################
    ################################## FEM ##################################
    #########################################################################

    # Optimization options for the form compiler
    parameters["form_compiler"]["cpp_optimize"] = True
    ffc_options = {"optimize": True, \
                "eliminate_zeros": True, \
                "precompute_basis_const": True, \
                "precompute_ip_const": True}

    # Create mesh and define function space
    rectangle = Rectangle(Point(0,0),Point(1.0, 1.0))
    circle = Circle(Point(center_coord), r0)
    region = rectangle - circle
    mesh = generate_mesh(region, 50)
    mesh_coord = mesh.coordinates()

    fig = plt.figure(constrained_layout=False, figsize=(5, 5))
    gs = fig.add_gridspec(1, 1)
    ## sigma_xx
    plot(mesh)
    fig.savefig('mesh.png')
    plt.close()

    exit()

    ## function space
    V = VectorFunctionSpace(mesh, "CG", 1)
    W = TensorFunctionSpace(mesh, "CG", 1)

    # Mark boundary subdomians
    bot =  CompiledSubDomain("near(x[1], side) && on_boundary", side = 0.0)
    top = CompiledSubDomain("near(x[1], side) && on_boundary", side = 1.0)
    interface = CompiledSubDomain(" sqrt(x[0]*x[0] + (x[1]-0.5)*(x[1]-0.5) ) <= 0.3+ 1e-6 && on_boundary")


    # Define Dirichlet boundary (x = 0 or x = 1)
    bot_disp = Expression(("0.0", "0.0"), degree=1)
    top_disp_exp = data2exp_2d(np.concatenate((top_coord_ref, top_disp), axis=1), V)
    bc1 = DirichletBC(V, bot_disp, bot)
    bc2 = DirichletBC(V, top_disp_exp, top)
    bcs = [bc1, bc2]

    ## Define inner traction 
    boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(0)
    intf = AutoSubDomain(lambda x: sqrt(x[0]*x[0]+(x[1]-0.5)*(x[1]-0.5)) < 0.3+1e-5)
    intf.mark(boundaries, 1)
    # mesh_intf = AutoSubDomain(lambda x: sqrt(x[0]*x[0]+x[1]*x[1]) < 0.3+1e-5)
    # mesh_intf.mark(boundaries, 2)
    ds_fem = ds(subdomain_data=boundaries)

    # Define functions
    du = TrialFunction(V) 
    v  = TestFunction(V)
    u  = Function(V) 
    B  = Constant((0.0, 0.0))
    # T_intf_exp = data2exp_2d(Txy_intf_true, V)
    T_intf_exp = Constant((0.0, 0.0))

    # Kinematics
    d = len(u)
    I = Identity(d)            
    F = I + grad(u)          
    C = F.T*F              
    b = F*F.T 

    # Invariants of deformation tensors
    Ic = tr(C)
    J  = det(F)

    # Elasticity parameters
    mu = E/(2*(1 + nu))
    lmbda = E*nu/((1 + nu)*(1 - 2*nu))
    K = E/(3*(1-2*nu))
    print(f'mu: {mu}, lmbda: {lmbda}, K: {K}')

    # Stored strain energy density (compressible neo-Hookean model)
    M1 = Constant((np.cos(alpha), np.sin(alpha)))
    M2 = Constant((np.cos(-alpha), np.sin(-alpha)))

    E1 = (1 - kappa/3)*(inner(C*M1, M1) - 1)
    E1 = (E1+abs(E1))/2.
    E2 = (1 - kappa/3)*(inner(C*M2, M2) - 1)
    E2 = (E2+abs(E2))/2.

    psi = (mu/2)*(Ic - 2) - mu*ln(J) + \
            (K1/K2/2)*(exp(K2*E1**2) - 1) + (K1/K2/2)*(exp(K2*E2**2) - 1) + \
            (K/2)*( (J**2 - 1)/2 - ln(J) )

    # Total potential energy
    Pi = psi*dx - dot(B, u)*dx + dot(T_intf_exp, u)*ds_fem(1)
    FF = derivative(Pi, u, v) 
    Jac = derivative(FF, u, du)

    # PK2 stress
    def PK2():
        return mu*(I - inv(C)) + 2*K1*E1*(exp(K2*E1**2) )*outer(M1, M1) + 2*K1*E2*(exp(K2*E2**2) )*outer(M2, M2) + \
        K/2*inv(C)*(J**2 - 1)

    #########################################################################
    ################################ DeepONet ###############################
    #########################################################################
    device = 'cuda'
    Tx_path = "./Tx_model/model_chkpts.pt"
    Ty_path = "./Ty_model/model_chkpts.pt"
   
    dim_br = [60, 80, 80, 80]
    dim_tr = [2, 40, 40, 80]
    
    model_Tx = opnn(dim_br, dim_tr).to(device)
    model_Tx = model_Tx.float()
    checkpoint = torch.load(Tx_path)
    model_Tx.load_state_dict(checkpoint["model_state_dict"])

    model_Ty = opnn(dim_br, dim_tr).to(device)
    model_Ty = model_Ty.float()
    checkpoint = torch.load(Ty_path)
    model_Ty.load_state_dict(checkpoint["model_state_dict"])

    ##################
    x_out = find_outside(mesh_coord, center_coord, r0)
    N_intf_out = circ_noraml(x_out, center_coord) ## assume circle
    x_t_tensor = torch.tensor(x_intf, dtype=torch.float).to(device) ## tensorize the input of the trunk net

    ## load true solution from Fenics
    # p11_fem_exp = Rbf(data_true[:, 0], data_true[:, 1], data_true[:, 4])
    # p12_fem_exp = Rbf(data_true[:, 0], data_true[:, 1], data_true[:, 5])
    # p21_fem_exp = Rbf(data_true[:, 0], data_true[:, 1], data_true[:, 6])
    # p22_fem_exp = Rbf(data_true[:, 0], data_true[:, 1], data_true[:, 7])
    
    # p11_e_mesh = p11_fem_exp(x_out[:, 0], x_out[:, 1])
    # p12_e_mesh = p12_fem_exp(x_out[:, 0], x_out[:, 1])
    # p21_e_mesh = p21_fem_exp(x_out[:, 0], x_out[:, 1])
    # p22_e_mesh = p22_fem_exp(x_out[:, 0], x_out[:, 1])
    u1_e_exp = Rbf(data_true[::intp, 0], data_true[::intp, 1], data_true[::intp, 2])
    u2_e_exp = Rbf(data_true[::intp, 0], data_true[::intp, 1], data_true[::intp, 3])
    
    u1_e = u1_e_exp(x_out[:, 0], x_out[:, 1])
    u2_e = u2_e_exp(x_out[:, 0], x_out[:, 1])

    u1_intf_e = u1_e_exp(x_intf[:, 0], x_intf[:, 1])
    u2_intf_e = u2_e_exp(x_intf[:, 0], x_intf[:, 1])
    intf_disp_exp = data2exp_2d(np.concatenate((x_intf, u1_intf_e[:, None], u2_intf_e[:, None]), axis=1), V)

    # bc3 = DirichletBC(V, intf_disp_exp, interface)
    # bcs = [bc1, bc2, bc3]

    #########################################################################
    ################################ Coupling ###############################
    #########################################################################
    err_his = []
    for iter in range(0, niter):

        ## FEM
        if iter == 0:
            solve(FF == 0, u, bcs, J=Jac, form_compiler_parameters=ffc_options)
        else:
            ## update T
            PK1 = F*PK2()
            PK1_out = projection(project(PK1, W), x_out).reshape(-1, 2, 2)    
            T_out = np.matmul(PK1_out, N_intf_out[:, :, None])[:, :, 0]

            Txy_out = np.concatenate((x_out, T_out), axis=1)
            Txy_intf = np.concatenate((x_intf, lmd/S_scaling), axis=1)
            Txy_tmp = np.concatenate((Txy_out, Txy_intf), axis=0) 
            Txy_intf_true = data2exp_2d(Txy_tmp, V)

            Pi = psi*dx - dot(B, u)*dx + dot(Txy_intf_true, u)*ds_fem(1)
            FF = derivative(Pi, u, v) 
            Jac = derivative(FF, u, du)
            solve(FF == 0, u, bcs, J=Jac, form_compiler_parameters=ffc_options)

        # PK1 = F*PK2()

        ## DeepONet
        u_intf = projection(u, x_intf)*U_scaling ## u from FEM
        x_b = np.concatenate((u_intf[:, 0], u_intf[:, 1]))
        # x_b_true = np.concatenate((u1_intf_e[None, :], u2_intf_e[None, :]), axis=1)*U_scaling


        ### plot u intf
        # fig = plt.figure(constrained_layout=False, figsize=(5, 5))
        # gs = fig.add_gridspec(1, 1)
        # ## sigma_xx
        # ax = fig.add_subplot(gs[0])
        # ax.plot(u_intf[:, 0]/U_scaling, label="Pred")
        # ax.plot(u1_intf_e, label="True")
        # ax.legend()
        # ax.set_title("prediction Tx")
        # fig.savefig('test_u.png')
        # plt.close()
        # ###############


        # # plot PK1 intf
        # PK1 = F*PK2()
        # pk1_intf = projection(project(PK1, W), x_intf) #*S_scaling
        # # print(pk1_intf)
        # Tx_intf_fem = pk1_intf[:, 0]*N_intf[:, 0] + pk1_intf[:, 1]*N_intf[:, 1]
        # Ty_intf_fem = pk1_intf[:, 2]*N_intf[:, 0] + pk1_intf[:, 3]*N_intf[:, 1]

        # #######
        # fig = plt.figure(constrained_layout=False, figsize=(10, 10))
        # gs = fig.add_gridspec(2, 2)
        # ## sigma_xx
        # ax = fig.add_subplot(gs[0])
        # ax.plot(pk1_intf[:, 0], label="Pred")
        # ax.plot(p11_intf, label="True")
        # ax = fig.add_subplot(gs[1])
        # ax.plot(pk1_intf[:, 1], label="Pred")
        # ax.plot(p12_intf, label="True")
        # ax = fig.add_subplot(gs[2])
        # ax.plot(pk1_intf[:, 2], label="Pred")
        # ax.plot(p21_intf, label="True")
        # ax = fig.add_subplot(gs[3])
        # ax.plot(pk1_intf[:, 3], label="Pred")
        # ax.plot(p22_intf, label="True")
        # ax.legend()
        # ax.set_title("prediction Tx")
        # fig.savefig('test_pk.png')
        # plt.close()

        # # traction
        # #######
        # fig = plt.figure(constrained_layout=False, figsize=(10, 5))
        # gs = fig.add_gridspec(1, 2)
        # ## sigma_xx
        # ax = fig.add_subplot(gs[0])
        # ax.plot(Tx_intf_fem, label="Pred")
        # ax.plot(Tx_intf_e, label="True")
        # ax = fig.add_subplot(gs[1])
        # ax.plot(Ty_intf_fem, label="Pred")
        # ax.plot(Ty_intf_e, label="True")
        # ax.legend()
        # ax.set_title("prediction Tx")
        # fig.savefig('test_tract.png')
        # exit()


        x_b = np.tile(x_b, (dim_intf, 1))
        x_b_tensor = torch.tensor(x_b, dtype=torch.float).to(device)

        Tx_intf_opnn = to_numpy(model_Tx.forward(x_b_tensor, x_t_tensor))
        Ty_intf_opnn = to_numpy(model_Ty.forward(x_b_tensor, x_t_tensor))
        Txy_intf = np.concatenate((Tx_intf_opnn, Ty_intf_opnn), axis=1)

        ################
        fig = plt.figure(constrained_layout=False, figsize=(10, 5))
        gs = fig.add_gridspec(1, 2)

        ## sigma_xx
        ax = fig.add_subplot(gs[0])
        ax.plot(Txy_intf[:, 0], label="Pred")
        ax.plot(Tx_intf_e*S_scaling, label="True")

        ax = fig.add_subplot(gs[1])
        ax.plot(Txy_intf[:, 1], label="Pred")
        ax.plot(Ty_intf_e*S_scaling, label="True")
        ax.legend()

        ax.set_title("prediction Ty")

        fig.savefig('test_opnn_traction.png')
        plt.close()
        ################


        ## traction on the interface from FEM
        PK1 = F*PK2()
        PK1_intf = projection(project(PK1, W), x_intf).reshape(-1, 2, 2) 
        lmd = np.matmul(PK1_intf, N_intf[:, :, None])[:, :, 0]
        lmd = lmd*S_scaling    

        ## dynamic theta
        # if iter == 0:
        #     s1_old = projection(project(sigma(u), W), x_intf)
        #     s2_old = np.concatenate((s11_intf, s12_intf, s12_intf, s22_intf), axis=1)/S_scaling
        #     # s1_old = projection(project(sigma(u), W), x_intf)[:, 0:1]
        #     # s2_old = s11_intf/S_scaling
        # else:
        #     s1_new = projection(project(sigma(u), W), x_intf)
        #     s2_new = np.concatenate((s11_intf, s12_intf, s12_intf, s22_intf), axis=1)/S_scaling
        #     # s1_new = projection(project(sigma(u), W), x_intf)[:, 0:1]
        #     # s2_new = s11_intf/S_scaling
        #     e1 = s1_new - s1_old
        #     e2 = s2_new - s2_old
        #     s1_old = np.copy(s1_new)
        #     s2_old = np.copy(s2_new)

        #     theta = (e1 * (e1 - e2)).sum()/((e1 - e2)**2).sum()
        #     print(theta)

        ## update lambda (traction in x, y direction)
        lmd = theta*Txy_intf + (1 - theta)*lmd

        ## error
        # p_fem_pred = projection(project(PK1, W), x_out)
        # err_1 = ((p_fem_pred[:, 0] - p11_e_mesh)**2).mean() + \
        #         ((p_fem_pred[:, 1] - p12_e_mesh)**2).mean() + \
        #         ((p_fem_pred[:, 2] - p21_e_mesh)**2).mean() + \
        #         ((p_fem_pred[:, 3] - p22_e_mesh)**2).mean()

        u_fem_intf = projection(u, x_intf)
        err_1 = ((u_fem_intf[:, 0] - u1_intf_e)**2).mean() + \
                ((u_fem_intf[:, 1] - u2_intf_e)**2).mean()
        err_2 = ((lmd[:, 0]/S_scaling - Tx_intf_e)**2 + (lmd[:, 1]/S_scaling - Ty_intf_e)**2).mean()

        # rela_err_1 = (abs((p_fem_pred[:, 0] - p11_e_mesh)/p11_e_mesh.max()) + \
        #         abs((p_fem_pred[:, 1] - p12_e_mesh)/p12_e_mesh.max()) + \
        #         abs((p_fem_pred[:, 2] - p21_e_mesh)/p21_e_mesh.max()) + \
        #         abs((p_fem_pred[:, 3] - p22_e_mesh)/p22_e_mesh.max()) ).mean()
        # rela_err_2 = abs((lmd/S_scaling - t_e)/t_e.max()).mean()

        err_his.append( np.array([err_1, err_2])) #, rela_err_1, rela_err_2]) )
        # print(f'iter: {iter}, err_1: {err_1:3g}, err_2: {err_2:3g}, rela_err_1: {rela_err_1}, rela_err_2: {rela_err_2}')
        print(f'iter: {iter}, err_1: {err_1:6g}, err_2: {err_2:6g}')


    ## save
    file_name = path + "/result.xdmf"
    file_results = XDMFFile(file_name)
    file_results.parameters["flush_output"] = True
    file_results.parameters["functions_share_mesh"] = True
    file_results.write(u, 0.)


    ## plot
    ## plot error
    err_his = np.array(err_his)
    fig = plt.figure(constrained_layout=False, figsize=(5, 5))
    gs = fig.add_gridspec(1, 1)
    ax = fig.add_subplot(gs[0])
    ax.plot(err_his[:, 0], label="err 1")
    ax.plot(err_his[:, 1], label="err 2")
    ax.set_yscale('log')
    ax.legend()
    ax.set_title("L2 square error")

    fig.savefig('error.png')
    plt.close()

    ## interfacial displacement
    fig = plt.figure(constrained_layout=False, figsize=(10, 5))
    gs = fig.add_gridspec(1, 2)
    ax = fig.add_subplot(gs[0])
    ax.plot(u_fem_intf[:, 0], label="ux fem")
    ax.plot(u1_intf_e, label="ux true")
    ax.legend()
    
    ax = fig.add_subplot(gs[1])
    ax.plot(u_fem_intf[:, 1], label="uy fem")
    ax.plot(u2_intf_e, label="uy true")
    ax.legend()

    fig.savefig('disp_final.png')
    plt.close()

    ## interfacial traction
    fig = plt.figure(constrained_layout=False, figsize=(10, 5))
    gs = fig.add_gridspec(1, 2)
    ax = fig.add_subplot(gs[0])
    ax.plot(lmd[:, 0]/S_scaling , label="Tx opnn")
    ax.plot(Tx_intf_e, label="Tx true")
    ax.legend()
    
    ax = fig.add_subplot(gs[1])
    ax.plot(lmd[:, 1]/S_scaling , label="Ty opnn")
    ax.plot(Ty_intf_e, label="Ty true")
    ax.legend()

    fig.savefig('traction_final.png')
    plt.close()


if __name__ == "__main__":
    main()
