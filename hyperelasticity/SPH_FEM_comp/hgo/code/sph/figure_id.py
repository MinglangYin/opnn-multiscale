import numpy as np
import matplotlib.pyplot as plt

## ITEM: ATOMS_id type x y z vx vy vz c_S[1] c_S[2] c_S[4] c_nn c_E[1] c_E[2] c_E[4] vx vy vz
data = np.loadtxt('SMD_id1863.dat')
data_2 = np.loadtxt('SMD_id1864.dat')

fig = plt.figure(constrained_layout=False, figsize=(5, 5))
gs = fig.add_gridspec(1, 1)    

ax = fig.add_subplot(gs[0])
#ax.plot(data[:, 0], 1.1*data[:, 0], 'b',linewidth=3, label='Analytical')
ax.plot(data[:, 3] - data[0, 3],  data[:, 9], '--r',linewidth=3, label='HGO Type')
ax.plot(data_2[:, 3] - data_2[0, 3],  data_2[:, 9], '--r',linewidth=3, label='HGO 1864 Type')

# ax.set_yscale('log')
# ax.set_xlim([0, epochs])
# ax.set_ylim([1, 4])
ax.set_xlabel('Strain')
ax.set_ylabel('Stress')
ax.legend()   

plt.tight_layout()
fig.savefig('./Results.png')
plt.close()    
