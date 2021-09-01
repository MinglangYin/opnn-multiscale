import numpy as np
import matplotlib.pyplot as plt

data_fn = np.loadtxt("stress_strain_nk_fenics.dat")
data_feb = np.loadtxt("stress_strain_nk_febio.dat")

fig = plt.figure(constrained_layout=False, figsize=(5, 5))
gs = fig.add_gridspec(1, 1)
ax = fig.add_subplot(gs[0])
h = ax.plot(data_fn[:, 4], data_fn[:, -1], label="Fenics", color = "blue", linewidth=3)
h = ax.plot(data_feb[:, 0], data_feb[:, 1], label="FEBio", color = "red", linestyle = "dashed", linewidth=3)
ax.set_xlim([0, 2.5])
ax.set_ylim([0, 3.5])
ax.set_xlabel("displacement")
ax.set_ylabel("s22")
ax.legend()
fig.savefig('str_disp.png')
plt.close()