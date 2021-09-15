import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('stress_strain.dat')

fig = plt.figure(constrained_layout=False, figsize=(5, 5))
gs = fig.add_gridspec(1, 1)    

ax = fig.add_subplot(gs[0])
#ax.plot(data[:, 0], 1.1*data[:, 0], 'b',linewidth=3, label='Analytical')
ax.plot(data[:, 0]-data[0, 0], data[:, 2], '--r',linewidth=3, label='HGO Type')

# ax.set_yscale('log')
# ax.set_xlim([0, epochs])
# ax.set_ylim([1, 4])
ax.set_xlabel('Strain')
ax.set_ylabel('Stress')
ax.legend()   

plt.tight_layout()
fig.savefig('./Results.png')
plt.close()    
