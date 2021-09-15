import numpy as np
import matplotlib.pyplot as plt

## ITEM: ATOMS_id type x y z vx vy vz c_S[1] c_S[2] c_S[4] c_nn c_E[1] c_E[2] c_E[4] vx vy vz
data_fn_cauchy = np.loadtxt('fenics_cauchy.dat')
data_sph_cauchy = np.loadtxt('sph_cauchy.dat')

fig = plt.figure(constrained_layout=False, figsize=(8, 8))
gs = fig.add_gridspec(2, 2)    

## fig 1: s11
ax = fig.add_subplot(gs[0])
ax.plot(data_fn_cauchy[:, 3], data_fn_cauchy[:, 4] , '--r',linewidth=3, label='fenics')
ax.plot(data_sph_cauchy[:, 3]-data_sph_cauchy[0, 3],  data_sph_cauchy[:, 5], 'orange',linewidth=3, label='sph')

## fig 2: s12
ax = fig.add_subplot(gs[1])
ax.plot(data_fn_cauchy[:, 3], data_fn_cauchy[:, 5] , '--r',linewidth=3, label='fenics')
ax.plot(data_sph_cauchy[:, 3]-data_sph_cauchy[0, 3],  data_sph_cauchy[:, 8], 'orange',linewidth=3, label='sph')

## fig 3: s13
ax = fig.add_subplot(gs[2])
ax.plot(data_fn_cauchy[:, 3], data_fn_cauchy[:, 6] , '--r',linewidth=3, label='fenics')
# ax.plot(data_sph_cauchy[:, 3]-data_sph_cauchy[0, 3],  data_sph_cauchy[:, 9], 'orange',linewidth=3, label='sph')

## fig 4: s21
ax = fig.add_subplot(gs[3])
ax.plot(data_fn_cauchy[:, 3], data_fn_cauchy[:, 7] , '--r',linewidth=3, label='fenics')
ax.plot(data_sph_cauchy[:, 3]-data_sph_cauchy[0, 3],  data_sph_cauchy[:, 6], 'orange',linewidth=3, label='sph')

ax.set_xlabel('Strain')
ax.set_ylabel('Stress')
ax.set_title("last")

ax.legend()   
plt.tight_layout()
fig.savefig('./Results.png')
plt.close()    
