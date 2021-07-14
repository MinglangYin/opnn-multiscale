import numpy as np
import matplotlib.pyplot as plt

from dolfin import *
from utils import *
# from mshr import *

class Zanolli(object):
    def __init__(self, dict_fem1, dict_fem2, eps = 0.001, norm = "L2"):
        ## FEM1
        self.model_fem = dict_fem1["model"]
        self.bc1_l = dict_fem1["bc_l"]
        self.bc1_r = dict_fem1["bc_r"] 
        self.x1 = dict_fem1["x"]
        self.V1 = dict_fem1["V"]

        ## FEM2
        self.model_fem = dict_fem2["model"]
        self.bc2_l = dict_fem2["bc_l"]
        self.bc2_r = dict_fem2["bc_r"] 
        self.x2 = dict_fem2["x"]
        self.V2 = dict_fem2["V"]

        self.x_intf = dict_fem1["x_intf"]

        ## initialization
        self.u_e1 = dict_fem1["u_e"]
        self.u1 = dict_fem1["u_e"]
        self.u_e_opnn = dict_fem2["u_e"]
        self.u_opnn = dict_fem2["u_e"]

        ##
        # self.u_fem1 = np.zeros((self.dim_g1, 1))

        ##
        self.lmd_old = 0.5
        self.lmd_new = 0.5
        self.theta = 0.5
        self.err_his = []
        if norm == "L2":
            self.error = self.l2_norm
    
    def l2_norm(self, u_e, u):
        return ((u_e - u)**2).mean()

    def run(self, niter, eps = 0.001, norm = "L2", **kwargs):
        self.eps = eps
        self.norm = "L2"

        for iter in range(0, niter):

            self.u_1 = self.model_fem1.forward([self.bc1_l, self.bc1_r])

            self.el.evaluate_basis_derivatives_all(2, values, x, coordinate_dofs, cell.orientation())


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