import numpy as np
import matplotlib.pyplot as plt

#data_fn = np.loadtxt("strain_stress_hgo_clamp_side_fenics.dat")
#data_feb = np.loadtxt("strain_stress_hgo_clamp_side_febio.dat")
#data_sph = np.loadtxt("strain_stress_hgo_clamp_side_sph.dat")
#data_sph = np.loadtxt("SMD_id1576.dat")

data_fn = np.loadtxt("strain_stress_hgo_clamp_top_fenics.dat")
data_feb = np.loadtxt("strain_stress_hgo_clamp_top_febio.dat")
data_sph = np.loadtxt("strain_stress_hgo_clamp_top_sph.dat")

fig = plt.figure(constrained_layout=False, figsize=(5, 5))
gs = fig.add_gridspec(1, 1)
ax = fig.add_subplot(gs[0])
h = ax.plot(data_fn[:, 4], data_fn[:, -1], label="Fenics", color = "blue", linewidth=3)
h = ax.plot(data_feb[:, 0], data_feb[:, 1], label="FEBio", color = "red", linestyle = "dashed", linewidth=3)
h = ax.scatter(data_sph[::10, 3] - data_sph[0, 3],  data_sph[::10, 9], marker="^", label="SPH", color = "black")
ax.set_xlim([0, 0.45])
ax.set_ylim([0, 7])
ax.set_xlabel("strain")
ax.set_ylabel("s22")
ax.legend()
fig.savefig('str_disp.png')
plt.close()
