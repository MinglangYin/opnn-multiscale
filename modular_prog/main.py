"""
Minglang Yin, minglang_yin@brown.edu
Multiscale Modeling: FEM + FEM
Schwarz alternating method
"""
import numpy as np
from utils import *
from solver import *
from coupling import *

def main():
    mesh1, mesh2, x_g1, x_g2, x_go, x_s, x_p = gen_mesh()

    ## FEM solver (domain s)
    u_true = Expression('1 + x[0]*x[0] + 2*x[1]*x[1]', degree=2)
    u_o = Expression('0', degree=1)
    V_s = FunctionSpace(mesh1, "P", 1)
    u_s = TrialFunction(V_s)
    v_s = TestFunction(V_s)

    f_s = Constant(-6.0)
    a_s = dot(grad(u_s), grad(v_s))*dx
    L_s = f_s*v_s*dx
    u_s = Function(V_s)
    model_fem_s = solver_fem(a_s, L_s, u_s)

    exp_go = 'on_boundary && x[0]>=0-DOLFIN_EPS && x[0]<=1+DOLFIN_EPS &&\
            x[1]>=0-DOLFIN_EPS && x[1]<=1+DOLFIN_EPS' # mesh_s outer boundary, need to be consistent
    exp_g1 = 'on_boundary && x[0]>=0.4-DOLFIN_EPS && x[0]<=0.6+DOLFIN_EPS &&\
            x[1]>=0.4-DOLFIN_EPS && x[1]<=0.6+DOLFIN_EPS' # mesh_s on Gemma 1
    bc_go = DirichletBC(V_s, u_true, exp_go)
    bc_g1 = DirichletBC(V_s, u_o, exp_g1)

    ## FEM solver (domain p)
    V_p = FunctionSpace(mesh2, "P", 1)
    u_es = interpolate(u_true, V_s)
    u_ep = interpolate(u_true, V_p)

    ## DeepONet
    device = "cuda"
    opnn_path = "./CheckPts/model_chkpts.pt"
    dim_in = 68
    dim_br = [dim_in, 75, 75, 75]
    dim_tr = [2, 75, 75, 75]
    model_opnn = opnn(dim_br, dim_tr).to(device)
    checkpoint = torch.load(opnn_path)
    model_opnn.load_state_dict(checkpoint["model_state_dict"])
    
    ## multiscale coupling
    dict_fem = {
        "model": model_fem_s,
        "bc_go": bc_go,
        "bc_g1": bc_g1,
        "x_go": x_go,
        "x_s": x_s,
        "u_es": u_es
    }
    dict_opnn = {
        "model": model_opnn,
        "x_g1": x_g1,
        "x_g2": x_g2, 
        "device": device,
        "x_p": x_p,
        "u_ep": u_ep,
        "dim_in": dim_in
    }

    MSC = Schwarz(dict_fem, dict_opnn, eps = 0.01, norm = "L2")
    MSC.run(niter = 20)
    # MSC.saveplot()

if __name__ == "__main__":
    main()