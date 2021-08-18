import numpy as np
import matplotlib.pyplot as plt

err_adp = np.loadtxt("err_his_adp.dat")
err_theta_025 = np.loadtxt("err_his_theta025.dat") 
err_theta_075 = np.loadtxt("err_his_theta075.dat") 
err_theta_05 = np.loadtxt("err_his_theta05.dat") 

## figures
fig = plt.figure(constrained_layout=False, figsize=(6, 6))
gs = fig.add_gridspec(1, 1)
ax = fig.add_subplot(gs[0])
ax.plot(err_adp[:, 2], color='red', label=r'Adaptive $\theta$ (FEM)')
ax.plot(err_adp[:, 3], color='red', linestyle = 'dashed', label=r'Adaptive $\theta$ (DeepONet)')
ax.plot(err_theta_025[:, 2], color='blue', label=r'$\theta$=0.25 (FEM)')
ax.plot(err_theta_025[:, 3], color='blue', linestyle = 'dashed', label=r'$\theta=0.25 (DeepONet)$')
ax.plot(err_theta_075[:, 2], color='black', label=r'$\theta$=0.75 (FEM)')
ax.plot(err_theta_075[:, 3], color='black',  linestyle = 'dashed', label=r'$\theta=0.75 (DeepONet)$')
ax.plot(err_theta_05[:, 2], color='orange', label=r'$\theta$=0.5 (FEM)')
ax.plot(err_theta_05[:, 3], color='orange', linestyle = 'dashed', label=r'$\theta=0.5$ (DeepONet)')

# ax.axhline(y=eps, linestyle='--', color='black',  label='$\epsilon$')

ax.set_xlabel('iter')
ax.set_ylabel('relative error')
ax.set_title('Convergence')
ax.set_yscale('log')
ax.legend()
plt.savefig("convergence.png")
plt.close()
