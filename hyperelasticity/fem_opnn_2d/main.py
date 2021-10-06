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


from dolfin import *
from mshr import *
from opnn import *
from utils import *

def main():
    path = 'Results'
    os.makedirs(path, exist_ok=True)

    niter = 10
    eps = 1e-5
    theta = 0.5
    r0 = 0.3

    U_scaling = 100 ## note that it is not confirmed yet!
    S_scaling = 10

    E, nu = 1.0, 0.3
    K1, K2 = 0.1, 1.5
    alpha = np.pi/2      
    kappa = 0

    ##################
    id_pick = 21

    top_disp_set = scipy.io.loadmat("hyper_notch_100_nofrac_topdisp.mat")
    top_disp = top_disp_set["disp_top"][id_pick]
    top_coord_ref = top_disp_set["coord_top_ref"][:, :2]

    dataset_sph = scipy.io.loadmat("hyper_notch_100_nofrac.mat")
    coord_init = dataset_sph["coord_init"]
    # snapshot = dataset["snapshot"]
    coord = dataset_sph["coord"]
    disp_sph = coord[id_pick] - coord_init
    data_true = np.concatenate((coord_init[:, :2], disp_sph[:, :2] ), axis=1)


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
    circle = Circle(Point(0.0, 0.5), r0)
    region = rectangle - circle
    mesh = generate_mesh(region, 25)
    mesh_coord = mesh.coordinates()

    ## function space
    V = VectorFunctionSpace(mesh, "CG", 1)
    W = TensorFunctionSpace(mesh, "CG", 1)

    # Mark boundary subdomians
    bot =  CompiledSubDomain("near(x[1], side) && on_boundary", side = 0.0)
    top = CompiledSubDomain("near(x[1], side) && on_boundary", side = 1.0)

    # Define Dirichlet boundary (x = 0 or x = 1)
    bot_disp = Expression(("0.0", "0.0"), degree=1)
    top_disp = data2exp_2d(np.concatenate((top_coord_ref, top_disp), axis=1), V)
    bc1 = DirichletBC(V, bot_disp, bot)
    bc2 = DirichletBC(V, top_disp, top)
    bcs = [bc1, bc2]

    ## Define inner traction 
    boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(0)
    intf = AutoSubDomain(lambda x: sqrt(x[0]*x[0]+(x[1]-0.5)*(x[1]-0.5)) < r0+1e-5)
    intf.mark(boundaries, 1)
    # mesh_intf = AutoSubDomain(lambda x: sqrt(x[0]*x[0]+x[1]*x[1]) < 0.3+1e-5)
    # mesh_intf.mark(boundaries, 2)
    ds_fem = ds(subdomain_data=boundaries)

    # Define functions
    du = TrialFunction(V) 
    v  = TestFunction(V)
    u  = Function(V) 
    B  = Constant((0.0, 0.0))
    T_intf = Constant((0.0, 0.0))

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
    Pi = psi*dx - dot(B, u)*dx - dot(T_intf, u)*ds_fem(1)
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
    P11_path = "./opnn_p11/model_chkpts.pt"
    P12_path = "./opnn_p12/model_chkpts.pt"
    P21_path = "./opnn_p21/model_chkpts.pt"
    P22_path = "./opnn_p22/model_chkpts.pt"

    dim_br = [60, 80, 80, 80]
    dim_tr = [2, 40, 40, 80]
    
    model_P11 = opnn(dim_br, dim_tr).to(device)
    model_P11 = model_P11.float()
    checkpoint = torch.load(P11_path)
    model_P11.load_state_dict(checkpoint["model_state_dict"])

    model_P12 = opnn(dim_br, dim_tr).to(device)
    model_P12 = model_P12.float()
    checkpoint = torch.load(P12_path)
    model_P12.load_state_dict(checkpoint["model_state_dict"])

    model_P21 = opnn(dim_br, dim_tr).to(device)
    model_P21 = model_P21.float()
    checkpoint = torch.load(P21_path)
    model_P21.load_state_dict(checkpoint["model_state_dict"])

    model_P22 = opnn(dim_br, dim_tr).to(device)
    model_P22 = model_P22.float()
    checkpoint = torch.load(P22_path)
    model_P22.load_state_dict(checkpoint["model_state_dict"])

    ## interface and lines
    x_intf = np.loadtxt("x_intf_ext.dat")
    center_coord = [0.0, 0.5]

    x_out = find_outside(mesh_coord, center_coord, r0)
    N_intf = circ_noraml(x_intf, center_coord) ## assume circle
    N_intf_out = circ_noraml(x_out, center_coord) ## assume circle
    x_t_tensor = torch.tensor(x_intf, dtype=torch.float).to(device) ## tensorize the input of the trunk net
    dim_intf = x_intf.shape[0]


    ## load true solution from Fenics
    # p11_fem_exp = Rbf(data_true[:, 0], data_true[:, 1], data_true[:, 4])
    # p12_fem_exp = Rbf(data_true[:, 0], data_true[:, 1], data_true[:, 5])
    # p21_fem_exp = Rbf(data_true[:, 0], data_true[:, 1], data_true[:, 6])
    # p22_fem_exp = Rbf(data_true[:, 0], data_true[:, 1], data_true[:, 7])
    
    # p11_e_mesh = p11_fem_exp(x_out[:, 0], x_out[:, 1])
    # p12_e_mesh = p12_fem_exp(x_out[:, 0], x_out[:, 1])
    # p21_e_mesh = p21_fem_exp(x_out[:, 0], x_out[:, 1])
    # p22_e_mesh = p22_fem_exp(x_out[:, 0], x_out[:, 1])
    u1_e_exp = Rbf(data_true[:, 0], data_true[:, 1], data_true[:, 2])
    u2_e_exp = Rbf(data_true[:, 0], data_true[:, 1], data_true[:, 3])
    
    u1_e = u1_e_exp(x_out[:, 0], x_out[:, 1])
    u2_e = u2_e_exp(x_out[:, 0], x_out[:, 1])


    #########################################################################
    ################################ Coupling ###############################
    #########################################################################
    err_his = []
    for iter in range(0, niter):
        PK1 = F*PK2()

        ## FEM
        if iter == 0:
            solve(FF == 0, u, bcs, J=Jac, form_compiler_parameters=ffc_options)
        else:
            ## update T
            PK1 = F*PK2()
            PK1_out = projection(project(PK1, W), x_out).reshape(-1, 2, 2)    
            T_out = np.matmul(PK1_out, N_intf_out[:, :, None])[:, :, 0]

            Txy_out = np.concatenate((x_out, -T_out), axis=1)
            Txy_intf = np.concatenate((x_intf, -lmd/S_scaling), axis=1)
            Txy_tmp = np.concatenate((Txy_out, Txy_intf), axis=0) 
            T_intf = data2exp_2d(Txy_tmp, V)

            Pi = psi*dx - dot(B, u)*dx - dot(T_intf, u)*ds_fem(1)
            FF = derivative(Pi, u, v) 
            Jac = derivative(FF, u, du)
            solve(FF == 0, u, bcs, J=Jac, form_compiler_parameters=ffc_options)

        ## DeepONet
        u_intf = projection(u, x_intf)*U_scaling
        x_b = u_intf.flatten()[None, :]
        x_b = np.tile(x_b, (dim_intf, 1))
        x_b_tensor = torch.tensor(x_b, dtype=torch.float).to(device)

        P11_intf = to_numpy(model_P11.forward(x_b_tensor, x_t_tensor))
        P12_intf = to_numpy(model_P12.forward(x_b_tensor, x_t_tensor))
        P21_intf = to_numpy(model_P21.forward(x_b_tensor, x_t_tensor))
        P22_intf = to_numpy(model_P21.forward(x_b_tensor, x_t_tensor))

        Tx_intf = P11_intf*N_intf[:, 0:1] + P12_intf*N_intf[:, 1:2]
        Ty_intf = P21_intf*N_intf[:, 0:1] + P22_intf*N_intf[:, 1:2]
        Txy_intf = np.concatenate((Tx_intf, Ty_intf), axis=1)

        ## traction on the interface from FEM
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

        u_fem = projection(u, x_out)
        err_1 = ((u_fem[:, 0] - u1_e)**2).mean() + \
                ((u_fem[:, 1] - u2_e)**2).mean()
        # err_2 = ((lmd/S_scaling - t_e)**2).mean()

        # rela_err_1 = (abs((p_fem_pred[:, 0] - p11_e_mesh)/p11_e_mesh.max()) + \
        #         abs((p_fem_pred[:, 1] - p12_e_mesh)/p12_e_mesh.max()) + \
        #         abs((p_fem_pred[:, 2] - p21_e_mesh)/p21_e_mesh.max()) + \
        #         abs((p_fem_pred[:, 3] - p22_e_mesh)/p22_e_mesh.max()) ).mean()
        # rela_err_2 = abs((lmd/S_scaling - t_e)/t_e.max()).mean()

        # err_his.append( np.array([err_1, err_2, rela_err_1, rela_err_2]) )
        # print(f'iter: {iter}, err_1: {err_1:3g}, err_2: {err_2:3g}, rela_err_1: {rela_err_1}, rela_err_2: {rela_err_2}')
        print(f'iter: {iter}, err_1: {err_1:3g}')#, rela_err_1: {rela_err_1}')

    ## save
    file_name = path + "/result.xdmf"
    file_results = XDMFFile(file_name)
    file_results.parameters["flush_output"] = True
    file_results.parameters["functions_share_mesh"] = True
    file_results.write(u, 0.)

if __name__ == "__main__":
    main()
