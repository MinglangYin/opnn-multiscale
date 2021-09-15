import numpy as np
import matplotlib.pyplot as plt

## ITEM: ATOMS_id type x y z vx vy vz c_S[1] c_S[2] c_S[4] c_nn c_E[1] c_E[2] c_E[4] vx vy vz
data_fn = np.loadtxt('fenics_ss_new.dat')
# data_sph_old = np.loadtxt('SMD_id1863_oldkernel.dat')
data_sph= np.loadtxt('SMD_id1863.dat')

fig = plt.figure(constrained_layout=False, figsize=(12, 12))
gs = fig.add_gridspec(3, 3)    

for i in range(0, 9):
    ax = fig.add_subplot(gs[i])
    ax.plot(data_fn[:, 4], data_fn[:, 6+i] , '--r',linewidth=3, label='fenics')
    ax.plot(data_sph[:, 3]-data_sph[0, 3],  data_sph[:, 5+i], 'b',linewidth=3, label='sph')
    #ax.plot(data_sph_old[:, 3]-data_sph_old[0, 3],  data_sph_old[:, 5+i], 'black',linewidth=3, label='sph old')
    ax.set_xlim([0, 0.5])
    #ax.set_ylim([0, 10])

ax.set_xlabel('Strain')
ax.set_ylabel('Stress')
ax.set_title("last")

ax.legend()   
plt.tight_layout()
fig.savefig('./Results.png')
plt.close()    
