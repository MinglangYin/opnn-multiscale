import numpy as np
import matplotlib.pyplot as plt

from dolfin import *
from utils import *
# from mshr import *

class Schwarz(object):
    def __init__(self, dict_fem, dict_opnn, eps = 0.001, norm = "L2"):
        ## FEM
        self.model_fem = dict_fem["model"]
        self.bc_go = dict_fem["bc_go"]
        self.bc_g1 = dict_fem["bc_g1"]
        self.x_go = dict_fem["x_go"] 
        self.x_s = dict_fem["x_s"]

        ## DeepONet
        self.model_opnn = dict_opnn["model"]
        self.x_g1 = dict_opnn["x_g1"]
        self.x_g2 = dict_opnn["x_g2"]
        self.x_p = dict_opnn["x_p"]
        device = dict_opnn["device"]
        dim_in = dict_opnn["dim_in"]
        dim_g1 = self.x_g1.shape[0]
        dim_g2 = self.x_g2.shape[0]
        
        self.opnn_br_g2 = np.zeros((dim_g1, dim_in))
        self.opnn_br_g1 = np.zeros((dim_g1, dim_in))
        self.x_s_intp = np.concatenate((self.x_g1, self.x_go), axis=0)

        ## initialization
        self.u_e_fem = dict_fem["u_es"] # true solution for domain s
        self.u_fem = dict_fem["u_es"]
        self.u_e_opnn = dict_opnn["u_ep"] # true solution for domain p
        self.u_opnn = dict_opnn["u_ep"] # interpolate solution from DeepONet

        ##
        self.u_fem_g1_old = np.zeros((self.dim_g1, 1))
        self.u_fem_g1_new = np.zeros((self.dim_g1, 1))
        self.u_fem_g2_old = np.zeros((self.dim_g2, 1))
        self.u_fem_g2_new = np.zeros((self.dim_g2, 1))
        self.u_opnn_g1_old = np.zeros((self.dim_g1, 1))
        self.u_opnn_g1_new = np.zeros((self.dim_g1, 1))
        self.u_opnn_g2_old = np.zeros((self.dim_g2, 1))
        self.u_opnn_g2_new = np.zeros((self.dim_g2, 1))

        ##
        self.tau_n1 = 0.5
        self.tau_n = 0.5
        self.eps = eps
        self.err_his = []
        if norm == "L2":
            self.error = self.l2_norm
    
    def l2_norm(self, u_e, u):
        return ((u_e - u)**2).mean()

    def run(self, niter, **kwargs):
        for iter in range(0, niter):
            if iter == 0:
                u_s = self.model_fem.forward([self.bc_go, self.bc_g1])
                exit()
                u_p_array = to_numpy(self.model_opnn.forward(self.opnn_br, self.x_p))
                # u_opnn = ## interpolate
                self.u_fem_g2_old = self.model_fem.interpolate(x_g2)
                self.u_opnn_g2_old = self.model_opnn.forward()
            else:
                ## step a
                self.u_opnn_g1_old = self.model_opnn.forward( )
                self.u_fem_g1_new = np.copy(self.u_opnn_g1_old)
                u_s_intp = np.concatenate((u_s_g1_new, u_s_out), axis=0)
                u_s_bc_exp = data2exp(np.concatenate((x_s_intp, u_s_intp), axis=1), V_s)

                ## step b
                self.bc_g1 = DirichletBC(V_s, u_s_bc_exp, self.bc_g1)
                self.model_fem([self.bc_go, self.bc_g1])

                ## step c, d
                self.u_opnn_g2_new = self.model_opnn.forward( )

                ## step e (relaxation)
                if iter == 1:
                    Q_n1 = np.copy(self.u_fem_g2_new)
                else:
                    Q_n1 = np.copy(Q_n)
                Q_n = self.u_fem_g2_new - self.u_opnn_g2_old
                
                self.tau_n = self.tau_n1 + (self.tau_n1 - 1)*((Q_n1 - Q_n)*Q_n).sum()/(((Q_n1 - Q_n)**2).sum())
                self.u_opnn_g2_new = self.tau_n*self.u_opnn_g2_old + (1 - self.tau_n)*self.u_opnn_g2_new

                ## step f
                u_p_stack = np.tile(self.u_opnn_g2_new.reshape(1, -1), [dim_p, 1])
                self.opnn_br_g1 = torch.tensor(u_p_stack, dtype=torch.float).to(device)
                self.u_p = to_numpy(model.forward(self.opnn_br_g1, self.x_p))

                ## update
                self.u_opnn_g2_old = np.copy(self.u_opnn_g2_new)
                self.tau_n1 = np.copy(self.tau_n)

            ## error 
            error_s = self.error(self.u_es, self.u_s)
            error_p = self.error(self.u_ep, self.u_p)
            self.err_his.append( np.array([error_s, error_p]) )
            print(f'iter: {iter}, error_s: {error_s:3g}, error_p: {error_p:3g}')

            if error_p <= self.eps and error_s <= self.eps:
                break
    
    # def saveplot(self):
    #     ## figures u_s
    #     fig = plt.figure(constrained_layout=False, figsize=(11, 3))
    #     gs = fig.add_gridspec(1, 3)
    #     ax = fig.add_subplot(gs[0])
    #     h = plot(u_s)
    #     plt.colorbar(h)
    #     ax.set_title('prediction')

    #     ax = fig.add_subplot(gs[1])
    #     h = plot(u_es)
    #     plt.colorbar(h)
    #     ax.set_title('true')

    #     ax = fig.add_subplot(gs[2])
    #     h = plot(abs(u_s - u_es))
    #     plt.colorbar(h)
    #     ax.set_title('err')

    #     plt.savefig("u_s.png")
    #     plt.close()

    #     ## figures u_p
    #     fig = plt.figure(constrained_layout=False, figsize=(11, 3))
    #     gs = fig.add_gridspec(1, 3)
    #     ax = fig.add_subplot(gs[0])
    #     h = plot(u_p_intp)
    #     plt.colorbar(h)
    #     ax.set_title('prediction')

    #     ax = fig.add_subplot(gs[1])
    #     h = plot(u_ep)
    #     plt.colorbar(h)
    #     ax.set_title('true')

    #     ax = fig.add_subplot(gs[2])
    #     h = plot(abs(u_p_intp - u_ep))
    #     plt.colorbar(h)
    #     ax.set_title('err')

    #     plt.savefig("u_p.png")
    #     plt.close()

    #     ## mesh 
    #     fig = plt.figure(constrained_layout=False, figsize=(11, 3))
    #     gs = fig.add_gridspec(1, 3)
    #     ax = fig.add_subplot(gs[0])
    #     plot(mesh_s, alpha=0.5)
    #     ax.set_xlim([0, 1])
    #     ax.set_ylim([0, 1])

    #     ax = fig.add_subplot(gs[1])
    #     plot(mesh_p, alpha=0.5)
    #     ax.set_xlim([0, 1])
    #     ax.set_ylim([0, 1])

    #     ax = fig.add_subplot(gs[2])
    #     plot(mesh_p, alpha=0.5)
    #     plot(mesh_s, alpha=0.5)
    #     ax.set_xlim([0, 1])
    #     ax.set_ylim([0, 1])

    #     plt.savefig("mesh.png")
    #     plt.close()

    #     ## figure convergence
    #     fig = plt.figure(constrained_layout=False, figsize=(4, 4))
    #     gs = fig.add_gridspec(1, 1)
    #     ax = fig.add_subplot(gs[0])
    #     ax.plot(err_his[:, 0], color='blue', label='$u_{s}$ (FEM)')
    #     ax.plot(err_his[:, 1], color='red', label='$u_{p}$ (DeepONet)')
    #     ax.axhline(y=eps, linestyle='--', color='black',  label='$\epsilon$')

    #     ax.set_xlabel('# iter')
    #     ax.set_ylabel('$L_{2}$ error')
    #     ax.set_yscale('log')
    #     ax.legend()
    #     plt.savefig("convergence.png")
    #     plt.close()