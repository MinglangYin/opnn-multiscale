import numpy as np
import matplotlib.pyplot as plt

# list = ['test_1', 'test_2', 'test_3', 'test_4', 'test_5', 'test_6', 'test_7', 'test_8']
# labels = ['Hg25', 'Hg50', 'q0.01', 'h3.01', 'h5.01', 'l0.02', 'q0.01', 'q0.001']
list = ['test_6', 'test_7', 'test_8']
labels = ['q0.06', 'q0.01', 'q0.001']

fig = plt.figure(constrained_layout=False, figsize=(12, 8))
gs = fig.add_gridspec(2, 3)    

## fig 1: s11
for ii, fname in enumerate(list):
    tmp = fname + '/SMD_id1690.dat'
    data_sph = np.loadtxt(tmp)
    for i in range(0, 6):
        print(i)
        ax = fig.add_subplot(gs[i])
        ax.plot(data_sph[:, 3] - data_sph[0, 3], data_sph[:, 5+i],linewidth=3, label=labels[ii])
        ax.set_xlabel('Strain')
        ax.set_ylabel('Stress')

data_fn = np.loadtxt("fenics_cauchy_ss.dat")
ax = fig.add_subplot(gs[0])
ax.plot(data_fn[:, 4], data_fn[:, 6],linewidth=3, color='black', linestyle='dashed', label="FEM")
ax.set_title('s11')

ax = fig.add_subplot(gs[1])
ax.plot(data_fn[:, 4], data_fn[:, 10],linewidth=3, color='black', linestyle='dashed', label="FEM")
ax.set_title('s22')

ax = fig.add_subplot(gs[2])
ax.plot(data_fn[:, 4], data_fn[:, 14],linewidth=3, color='black', linestyle='dashed', label="FEM")
ax.set_title('s33')

ax = fig.add_subplot(gs[3])
ax.plot(data_fn[:, 4], data_fn[:, 7],linewidth=3, color='black', linestyle='dashed', label="FEM")
ax.set_title('s12')

ax = fig.add_subplot(gs[4])
ax.plot(data_fn[:, 4], data_fn[:, 8],linewidth=3, color='black', linestyle='dashed', label="FEM")
ax.set_title('s13')

ax = fig.add_subplot(gs[5])
ax.plot(data_fn[:, 4], data_fn[:, 11],linewidth=3, color='black', linestyle='dashed', label="FEM")
ax.set_title('s23')


ax.legend()   
plt.tight_layout()
fig.savefig('./comp.png')
plt.close()    
