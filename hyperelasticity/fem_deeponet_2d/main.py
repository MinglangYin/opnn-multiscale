"""
## Author: Minglang Yin, minglang_yin@brown.edu
## Uniaxial pulling of a 3D plate
## Holzapfel-Gasser-Odgen Model
## Clamped bottom
## Impose displacement BC on the top
"""
import os
import numpy as np

from dolfin import *
from mshr import *
from opnn import *
from utils import *

def find_outside(x, center_coord, r0):
    list = []
    x0, x1 = center_coord[0], center_coord[1]
    for pt in x:
        if ( (pt[0]-x0)**2+(pt[1]-x1)**2) > r0**2:
            list.append(pt)
    return np.array(list)

def circ_noraml(x, center):
    norm = np.linalg.norm(x-center, ord=2, axis=1)[:, None]
    return (x-center)/norm

def main():
    # make new directory
    path = 'Results'
    os.makedirs(path, exist_ok=True)

    niter = 2
    eps = 1e-5
    theta = 0.5
    r0 = 0.3

    U_scaling = 10 ## note that it is not confirmed yet!
    S_scaling = 100

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
    box = Box(Point(0,0,0),Point(1.0, 1.0, 0.05))
    cylinder = Cylinder(Point(0.0, 0.5, 0.05), Point(0.0, 0.5, 0.0), r0, r0)
    region = box - cylinder
    mesh = generate_mesh(region, 25)
    mesh_coord = mesh.coordinates()
    
    ## function space
    V = VectorFunctionSpace(mesh, "CG", 1)
    W = TensorFunctionSpace(mesh, "CG", 1)

    # Mark boundary subdomians
    bot =  CompiledSubDomain("near(x[1], side) && on_boundary", side = 0.0)
    top = CompiledSubDomain("near(x[1], side) && on_boundary", side = 1.0)

    # Define Dirichlet boundary (x = 0 or x = 1)
    bot_disp = Expression(("0.0", "0.0", "0.0"), degree=1)
    top_disp = Expression(("t*0.0","t*0.1", "t*0.0"), t = 1, degree=1)
    bc1 = DirichletBC(V, bot_disp, bot)
    bc2 = DirichletBC(V, top_disp, top)
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
    du = TrialFunction(V)            # Incremental displacement
    v  = TestFunction(V)             # Test function
    u  = Function(V)                 # Displacement from previous iteration
    B  = Constant((0.0, 0.0, 0.0))   # Body force per unit volume
    T_intf = Constant((0.0, 0.0, 0.0))

    # Kinematics
    d = len(u)
    I = Identity(d)             # Identity tensor
    F = I + grad(u)             # Deformation gradient
    C = F.T*F                   # Right Cauchy-Green tensor
    b = F*F.T                   # Left Cauchy-Green tensor

    # Invariants of deformation tensors
    Ic = tr(C)
    J  = det(F)

    # Elasticity parameters
    E, nu = 1.0, 0.3
    mu, lmbda = E/(2*(1 + nu)), E*nu/((1 + nu)*(1 - 2*nu))
    K = E/(3*(1-2*nu)) # kappa

    print(f'mu: {mu}, lmbda: {lmbda}, K: {K}')

    # Stored strain energy density (compressible neo-Hookean model)
    K1, K2 = 0.1, 1.5
    alpha = np.pi/2      
    kappa = 0
    M1 = Constant((np.cos(alpha), np.sin(alpha), 0))
    M2 = Constant((np.cos(-alpha), np.sin(-alpha), 0))

    E1 = (1 - kappa/3)*(inner(C*M1, M1) - 1)
    E1 = (E1+abs(E1))/2.

    E2 = (1 - kappa/3)*(inner(C*M2, M2) - 1)
    E2 = (E2+abs(E2))/2.

    psi = (mu/2)*(Ic - 3) - mu*ln(J) + \
            (K1/K2/2)*(exp(K2*E1**2) - 1) + (K1/K2/2)*(exp(K2*E2**2) - 1) + \
            (K/2)*( (J**2 - 1)/2 - ln(J) )

    # Total potential energy
    Pi = psi*dx - dot(B, u)*dx - dot(T_intf, u)*ds_fem(1)
    FF = derivative(Pi, u, v)      # Compute first variation of Pi (directional derivative about u in the direction of v)
    Jac = derivative(FF, u, du)      # Compute Jacobian of F

    # PK2 stress
    def PK2():
        return mu*(I - inv(C)) + 2*K1*E1*(exp(K2*E1**2) )*outer(M1, M1) + 2*K1*E2*(exp(K2*E2**2) )*outer(M2, M2) + \
        K/2*inv(C)*(J**2 - 1)

    #########################################################################
    ################################ DeepONet ###############################
    #########################################################################
    device = 'cuda'
    P11_path = "./CheckPts/model_chkpts.pt" # the model at this stage is not correct
    P12_path = "./CheckPts/model_chkpts.pt"
    P21_path = "./CheckPts/model_chkpts.pt"
    P22_path = "./CheckPts/model_chkpts.pt"

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
    N_intf_out = circ_noraml(x_out[:, 0:2], center_coord) ## assume circle
    x_t_tensor = torch.tensor(x_intf, dtype=torch.float).to(device) ## tensorize the input of the trunk net
    dim_intf = x_intf.shape[0]
    zeros = np.zeros((x_intf.shape[0], 1))

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
            # PK1 = F*PK2()
            PK1_out = projection(project(PK1, W), x_out).reshape(-1, 3, 3)     ## stress from FEM
            T_out = np.matmul(PK1_out, np.concatenate((N_intf_out, np.zeros((x_out.shape[0], 1))), axis=1)[:, :, None])[:, :, 0]  ## N_intf pointing toward outside   

            Txy_out = np.concatenate((x_out, -T_out), axis=1)        ## need to turn t_out input pointing inward
            Txy_intf = np.concatenate((np.concatenate((x_intf, zeros), axis=1), -lmd/S_scaling), axis=1) ## need to rescale, pointing toward inside
            Txy_tmp = np.concatenate((Txy_out, Txy_intf), axis=0) 
            T_intf = data2exp_3d(Txy_tmp, V)

            solve(FF == 0, u, bcs, J=Jac, form_compiler_parameters=ffc_options)

        ## DeepONet
        
        u_intf = projection(u, np.concatenate((x_intf, zeros), axis=1) )*U_scaling
        x_b = u_intf[:, 0:2].flatten()[None, :] # only take disp in x and y
        x_b = np.tile(x_b, (dim_intf, 1))
        x_b_tensor = torch.tensor(x_b, dtype=torch.float).to(device)

        P11_intf = to_numpy(model_P11.forward(x_b_tensor, x_t_tensor))        ## PK1 11
        P12_intf = to_numpy(model_P12.forward(x_b_tensor, x_t_tensor))        ## PK1 12
        P21_intf = to_numpy(model_P21.forward(x_b_tensor, x_t_tensor))        ## PK1 12
        P22_intf = to_numpy(model_P21.forward(x_b_tensor, x_t_tensor))        ## PK1 22

        Tx_intf = P11_intf*N_intf[:, 0:1] + P12_intf*N_intf[:, 1:2]
        Ty_intf = P21_intf*N_intf[:, 0:1] + P22_intf*N_intf[:, 1:2]
        Txy_intf = np.concatenate((Tx_intf, Ty_intf, zeros), axis=1)

        ## traction on the interface from FEM
        PK1_intf = projection(project(PK1, W), np.concatenate((x_intf, zeros), axis=1)).reshape(-1, 3, 3)     ## stress from FEM
        lmd = np.matmul(PK1_intf, np.concatenate((N_intf, zeros), axis=1)[:, :, None])[:, :, 0]               ## n_intf pointing toward outside
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
        s_fem_pred = projection(project(PK1, W), mesh_coord)

        err_1 = ((s_fem_pred[:, 0])**2).mean() + \
                ((s_fem_pred[:, 1])**2).mean() + \
                ((s_fem_pred[:, 3])**2).mean()
        err_2 = ((lmd/S_scaling)**2).mean()

        # rela_err_1 = (abs(s_fem_pred[:, 0]) + abs(s_fem_pred[:, 1]) + abs(s_fem_pred[:, 3])).mean()
        # rela_err_2 = abs((lmd/S_scaling)).mean()

        err_his.append( np.array([err_1, err_2]) )
        print(f'iter: {iter}, err_1: {err_1:3g}, err_2: {err_2:3g}')

        ## need true solution
        # err_1 = ((s_fem_pred[:, 0] - s11_e_mesh)**2).mean() + \
        #         ((s_fem_pred[:, 1] - s12_e_mesh)**2).mean() + \
        #         ((s_fem_pred[:, 3] - s22_e_mesh)**2).mean()
        # err_2 = ((lmd/S_scaling - t_e)**2).mean()

        # rela_err_1 = (abs((s_fem_pred[:, 0] - s11_e_mesh)/s11_e_mesh.max()) + \
        #         abs((s_fem_pred[:, 1] - s12_e_mesh)/s12_e_mesh.max()) + \
        #         abs((s_fem_pred[:, 3] - s22_e_mesh)/s22_e_mesh.max()) ).mean()*1/3
        # rela_err_2 = abs((lmd/S_scaling - t_e)/t_e.max()).mean()

        # err_his.append( np.array([err_1, err_2, rela_err_1, rela_err_2]) )
        # print(f'iter: {iter}, err_1: {err_1:3g}, err_2: {err_2:3g}, rela_err_1: {rela_err_1}, rela_err_2: {rela_err_2}')

    ## save
    file_name = path + "/result.xdmf"
    file_results = XDMFFile(file_name)
    file_results.parameters["flush_output"] = True
    file_results.parameters["functions_share_mesh"] = True
    file_results.write(u, 0.)

if __name__ == "__main__":
    main()