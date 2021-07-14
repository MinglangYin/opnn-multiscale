import numpy as np
import matplotlib.pyplot as plt
# import torch

from dolfin import *
from utils import *
# from mshr import *

class Schwarz(object):
    def __init__(self, dict_fem, dict_opnn):
        ## FEM
        self.model_fem = dict_fem["model"]
        self.bc_go = dict_fem["bc_go"]
        self.bc_g1 = dict_fem["bc_g1"]
        self.x_go = dict_fem["x_go"] 
        self.x_s = dict_fem["x_s"]
        self.u_es = dict_fem["u_es"] ## true solution
        self.u_s = dict_fem["u_es"] ## FEM solution
        # self.u_s_arr = np.array(self.u_s.vector())
        self.V_s = dict_fem["V_s"]
        self.mesh_s = dict_fem["mesh_s"]
        self.exp_g1 = dict_fem["exp_g1"]

        ## DeepONet
        self.model_opnn = dict_opnn["model"]
        self.x_g1 = dict_opnn["x_g1"]
        self.x_g2 = dict_opnn["x_g2"]
        self.x_p = dict_opnn["x_p"]
        self.V_p = dict_opnn["V_p"]
        self.u_ep = dict_opnn["u_ep"] ## true solution
        self.u_p = dict_opnn["u_ep"] # interpolate solution from DeepONet
        self.mesh_p = dict_opnn["mesh_p"]

        self.device = dict_opnn["device"]
        self.dim_in = dict_opnn["dim_in"]
        self.dim_g1 = self.x_g1.shape[0]
        self.dim_g2 = self.x_g2.shape[0]
        self.dim_p = self.x_p.shape[0]
        
        self.opnn_br_g1 = np.zeros((self.dim_g1, self.dim_in))
        self.opnn_br_g2 = np.zeros((self.dim_g2, self.dim_in))
        self.opnn_br_p = np.zeros((self.dim_p, self.dim_in))
        # self.x_s_intp = np.concatenate((self.x_g1, self.x_s), axis=0)

        self.x_p_tensor = torch.tensor(self.x_p, dtype=torch.float).to(self.device)
        self.x_g1_tensor = torch.tensor(self.x_g1, dtype=torch.float).to(self.device)
        self.x_g2_tensor = torch.tensor(self.x_g2, dtype=torch.float).to(self.device)
        self.opnn_br_p_tensor = torch.tensor(self.opnn_br_p, dtype=torch.float).to(self.device)
        self.opnn_br_g1_tensor = torch.tensor(self.opnn_br_g1, dtype=torch.float).to(self.device)
        self.opnn_br_g2_tensor = torch.tensor(self.opnn_br_g2, dtype=torch.float).to(self.device)
        
        ## initialization
        self.u_s_g1_old = np.zeros((self.dim_g1, 1))
        self.u_s_g1_new = np.zeros((self.dim_g1, 1))
        self.u_s_g2_old = np.zeros((self.dim_g2, 1))
        self.u_s_g2_new = np.zeros((self.dim_g2, 1))
        self.u_p_g1_old = np.zeros((self.dim_g1, 1))
        self.u_p_g1_new = np.zeros((self.dim_g1, 1))
        self.u_p_g2_old = np.zeros((self.dim_g2, 1))
        self.u_p_g2_new = np.zeros((self.dim_g2, 1))

        ##
        self.tau_n1 = 0.5
        self.tau_n = 0.5
        self.err_his = []
    
    def l2_norm(self, u_e, u):
        return (( np.array(u_e.vector()) - np.array(u.vector()) )**2).mean()

    def _interp(self, x, u, V):
        u_intp_exp = data2exp(np.concatenate((x, u), axis=1), V)
        return interpolate(u_intp_exp, V)


    def get_true_u(self, coord):
        return 1+coord[:, 0]**2+2*coord[:, 1]**2

    def _tile(self, u, name):
        u = u.reshape(1, -1)
        if name == "g1":
            dim = self.dim_g1
        elif name == "g2":
            dim = self.dim_g2
        elif name == "p":
            dim = self.dim_p
        u_t = np.tile(u, [dim, 1])
        return torch.tensor(u_t, dtype=torch.float).to(self.device)

    def run(self, niter, eps = 0.001, norm = "L2", **kwargs):
        self.eps = eps
        self.norm = "L2"
        u_s_out = self.get_true_u(self.x_go)[:, None]
        for iter in range(0, niter):
            if iter == 0:
                self.u_s = self.model_fem.forward([self.bc_go, self.bc_g1])
                u_p_arr = to_numpy(self.model_opnn.forward(self.opnn_br_p_tensor, self.x_p_tensor))
                
                ## updates
                self.u_p = self._interp(self.x_p, u_p_arr, self.V_p) 
                self.u_s_g2_old = self.model_fem.interpolate(self.x_g2)
                self.u_p_g2_old = to_numpy(self.model_opnn.forward(self.opnn_br_g2_tensor, self.x_g2_tensor))
                u_s_arr = np.array(self.u_s.vector())[:, None]
            else:
                ## step a
                self.opnn_br_tensor = self._tile(self.u_p_g2_old, "g1")
                self.u_s_g1_new = to_numpy(self.model_opnn.forward(self.opnn_br_tensor, self.x_g1_tensor))

                # u_s_intp = np.concatenate((self.u_s_g1_new, u_s_arr), axis=0)
                u_s_intp = np.concatenate((self.u_s_g1_new, u_s_out), axis=0)
                x_s_intp = np.concatenate((self.x_g1, self.x_go), axis=0)
                u_s_bc_exp = data2exp(np.concatenate((x_s_intp, u_s_intp), axis=1), self.V_s)

                ## step b
                self.bc_g1 = DirichletBC(self.V_s, u_s_bc_exp, self.exp_g1)
                self.u_s = self.model_fem.forward([self.bc_go, self.bc_g1])

                ## step c, d
                self.u_s_g2_new = self.model_fem.interpolate(self.x_g2)

                ## step e (relaxation)
                if iter == 1:
                    Q_n1 = np.copy(self.u_s_g2_new)
                else:
                    Q_n1 = np.copy(Q_n)
                Q_n = self.u_s_g2_new - self.u_p_g2_old
                
                self.tau_n = self.tau_n1 + (self.tau_n1 - 1)*((Q_n1 - Q_n)*Q_n).sum()/(((Q_n1 - Q_n)**2).sum())
                self.u_p_g2_new = self.tau_n*self.u_p_g2_old + (1 - self.tau_n)*self.u_s_g2_new

                ## step f
                self.opnn_br_tensor = self._tile(self.u_p_g2_new, "p")
                u_p_arr = to_numpy(self.model_opnn.forward(self.opnn_br_tensor, self.x_p_tensor))
                self.u_p = self._interp(self.x_p, u_p_arr, self.V_p) 

                ## update
                self.u_p_g2_old = np.copy(self.u_p_g2_new)
                self.tau_n1 = np.copy(self.tau_n)
                u_s_arr = np.array(self.u_s.vector())[:, None]

            ## error 
            if norm == "L2":
                error_s = self.l2_norm(self.u_es, self.u_s)
                error_p = self.l2_norm(self.u_ep, self.u_p)
            self.err_his.append( np.array([error_s, error_p]) )
            print(f'iter: {iter}, error_s: {error_s:3g}, error_p: {error_p:3g}')

            if error_p <= eps and error_s <= eps:
                break
        
        self.err_his = np.array(self.err_his)
    
    def saveplot(self):
        ##### figures 1: u_s #####
        fig = plt.figure(constrained_layout=False, figsize=(11, 3))
        gs = fig.add_gridspec(1, 3)
        ax = fig.add_subplot(gs[0])
        h = plot(self.u_s)
        plt.colorbar(h)
        ax.set_title('prediction')

        ax = fig.add_subplot(gs[1])
        h = plot(self.u_es)
        plt.colorbar(h)
        ax.set_title('true')

        ax = fig.add_subplot(gs[2])
        h = plot(abs(self.u_s - self.u_es))
        plt.colorbar(h)
        ax.set_title('err')

        plt.savefig("u_s.png")
        plt.close()

        ##### figures 2: u_p #####
        fig = plt.figure(constrained_layout=False, figsize=(11, 3))
        gs = fig.add_gridspec(1, 3)
        ax = fig.add_subplot(gs[0])
        h = plot(self.u_p)
        plt.colorbar(h)
        ax.set_title('prediction')

        ax = fig.add_subplot(gs[1])
        h = plot(self.u_ep)
        plt.colorbar(h)
        ax.set_title('true')

        ax = fig.add_subplot(gs[2])
        h = plot(abs(self.u_p - self.u_ep))
        plt.colorbar(h)
        ax.set_title('err')

        plt.savefig("u_p.png")
        plt.close()

        ##### figures 3: mesh #####
        fig = plt.figure(constrained_layout=False, figsize=(11, 3))
        gs = fig.add_gridspec(1, 3)
        ax = fig.add_subplot(gs[0])
        plot(self.mesh_s, alpha=0.5)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_title("mesh s")

        ax = fig.add_subplot(gs[1])
        plot(self.mesh_p, alpha=0.5)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_title("mesh p")

        ax = fig.add_subplot(gs[2])
        plot(self.mesh_p, alpha=0.5)
        plot(self.mesh_s, alpha=0.5)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_title("mesh s+p")

        plt.savefig("mesh.png")
        plt.close()

        ##### figures 4: convergence #####
        fig = plt.figure(constrained_layout=False, figsize=(4, 4))
        gs = fig.add_gridspec(1, 1)
        ax = fig.add_subplot(gs[0])
        ax.plot(self.err_his[:, 0], color='blue', label='$u_{s}$ (FEM)')
        ax.plot(self.err_his[:, 1], color='red', label='$u_{p}$ (DeepONet)')
        ax.axhline(y=self.eps, linestyle='--', color='black',  label='$\epsilon$')

        ax.set_xlabel('# iter')
        ax.set_ylabel('$L_{2}$ error')
        ax.set_yscale('log')
        ax.legend()
        plt.savefig("convergence.png")
        plt.close()