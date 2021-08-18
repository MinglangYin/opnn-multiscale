import numpy as np
import matplotlib.pyplot as plt

steps = 40
aa = 1
bb = 1.25
path = "outline_f0.08/"
for i in range(0, steps):
    f_in = path + "iter_" + str(i) + "_f_in.txt"
    f_out = path + "iter_" + str(i) + "_f_out.txt"
    f_step = path + "iter_" + str(i) + ".png"
    f_in_true = path + "f_in_e.txt"
    f_out_true = path + "f_out_e.txt"

    line_in = np.loadtxt(f_in)
    line_out = np.loadtxt(f_out)
    line_in_e = np.loadtxt(f_in_true)
    line_out_e = np.loadtxt(f_out_true)

    fig = plt.figure(constrained_layout=False, figsize=(4, 4))
    gs = fig.add_gridspec(1, 1)

    ax = fig.add_subplot(gs[0])
    h = ax.plot(line_in_e[:, 0] + aa*line_in_e[:, 2], line_in_e[:, 1] + bb*line_in_e[:, 3], color = "blue")
    h = ax.plot(line_out_e[:, 0] + aa*line_out_e[:, 2], line_out_e[:, 1] + bb*line_out_e[:, 3], color = "blue")
    h = ax.plot(line_in[:, 0] + aa*line_in[:, 2], line_in[:, 1] + bb*line_in[:, 3], color = "red", linestyle="dashed")
    h = ax.plot(line_out[:, 0] + aa*line_out[:, 2], line_out[:, 1] + bb*line_out[:, 3], color = "red", linestyle="dashed")
    ax.axis("off")
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])

    fig.savefig(f_step, transparent=True)
    plt.close()

err_his = np.loadtxt("err_his_theta05.dat")

## figures
fig = plt.figure(constrained_layout=False, figsize=(5.5, 4.5))
gs = fig.add_gridspec(1, 1)
ax = fig.add_subplot(gs[0])
ax.plot(err_his[:, 2], color='black', label=r'$\theta$=0.5 (FEM)')
ax.plot(err_his[:, 3], color='black', linestyle = 'dashed', label=r'$\theta$=0.5 (DeepONet)')

ax.set_xlabel('iter')
ax.set_ylabel('relative error')
# ax.set_ylim([1e-7, 1e-1])
ax.set_yscale('log')
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
# ax.legend()
plt.savefig("convergence.png")
plt.close()
