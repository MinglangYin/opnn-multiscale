import numpy as np
import matplotlib.pyplot as plt

err_dyn = np.loadtxt("err_dyn.dat")
err_025 = np.loadtxt("err_025.dat")
err_050 = np.loadtxt("err_050.dat")
err_075 = np.loadtxt("err_075.dat")

fig = plt.figure(constrained_layout=False, figsize=(4, 4))
gs = fig.add_gridspec(1, 1)
ax = fig.add_subplot(gs[0])
ax.plot(err_dyn[:, 0], color='blue', label='dyn')
ax.plot(err_025[:, 0], color='red', label=r'$\theta$=0.25')
# ax.plot(err_050[:, 0], color='black', label=r'$\theta$=0.50')
# ax.plot(err_075[:, 0], color='green', label=r'$\theta$=0.75')

ax.set_xlabel('# iter')
ax.set_ylabel('$L_{2}$ error')
ax.set_yscale('log')
ax.legend()
plt.savefig("convergence.png")
plt.close()
