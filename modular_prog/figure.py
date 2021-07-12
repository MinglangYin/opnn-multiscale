import numpy as np
import matplotlib.pyplot as plt

from dolfin import *
from mshr import *
from scipy.interpolate import interp2d

def post_figure(Omega):

    ## figures u_s
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

    ## figures u_p
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

    ## mesh 
    fig = plt.figure(constrained_layout=False, figsize=(6, 6))
    gs = fig.add_gridspec(1, 1)
    ax = fig.add_subplot(gs[0])
    plot(mesh_s, alpha=0.5)
    plot(mesh_p, alpha=0.5)
    plt.savefig("mesh.png")
    plt.close()

    ## figure convergence
    fig = plt.figure(constrained_layout=False, figsize=(4, 4))
    gs = fig.add_gridspec(1, 1)
    ax = fig.add_subplot(gs[0])
    ax.plot(err_his[:, 0], color='blue', label='$u_{s}$ (FEM)')
    ax.plot(err_his[:, 1], color='red', label='$u_{p}$ (FEM)')
    ax.axhline(y=eps, linestyle='--', color='black',  label='$\epsilon$')

    ax.set_yscale('log')
    ax.set_xlabel('# iter')
    ax.set_ylabel('$L_{2}$ error')
    ax.legend()
    plt.savefig("convergence.png")
    plt.close()