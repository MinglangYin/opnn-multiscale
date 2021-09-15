import numpy as np
import matplotlib.pyplot as plt

## ITEM: ATOMS_id type x y z vx vy vz c_S[1] c_S[2] c_S[4] c_nn c_E[1] c_E[2] c_E[4] vx vy vz
data_fn_cauchy = np.loadtxt('stress_strain.dat')
#data_sph_cauchy = np.loadtxt('sph_cauchy_ss.dat')
data_sph_cauchy_new = np.loadtxt('SMD_id2691.dat')

fig = plt.figure(constrained_layout=False, figsize=(12, 12))
gs = fig.add_gridspec(3, 3)    

## fig 1: s11
ax = fig.add_subplot(gs[0])
ax.plot(data_fn_cauchy[:, 4], data_fn_cauchy[:, 6] , '--r',linewidth=3, label='fenics')
#ax.plot(data_sph_cauchy[:, 3]-data_sph_cauchy[0, 3],  data_sph_cauchy[:, 5], 'b',linewidth=3, label='sph')
ax.plot(data_sph_cauchy_new[:, 3]-data_sph_cauchy_new[0, 3],  data_sph_cauchy_new[:, 5], 'orange',linewidth=3, label='sph')

## fig 2: s12
ax = fig.add_subplot(gs[1])
ax.plot(data_fn_cauchy[:, 4], data_fn_cauchy[:, 7] , '--r',linewidth=3, label='fenics')
#ax.plot(data_sph_cauchy[:, 3]-data_sph_cauchy[0, 3],  data_sph_cauchy[:, 8], 'b',linewidth=3, label='sph')
ax.plot(data_sph_cauchy_new[:, 3]-data_sph_cauchy_new[0, 3],  data_sph_cauchy_new[:, 8], 'orange',linewidth=3, label='sph')

## fig 3: s13
ax = fig.add_subplot(gs[2])
ax.plot(data_fn_cauchy[:, 4], data_fn_cauchy[:, 8] , '--r',linewidth=3, label='fenics')
#ax.plot(data_sph_cauchy[:, 3]-data_sph_cauchy[0, 3],  data_sph_cauchy[:, 9], 'b',linewidth=3, label='sph')
ax.plot(data_sph_cauchy_new[:, 3]-data_sph_cauchy_new[0, 3],  data_sph_cauchy_new[:, 9], 'orange',linewidth=3, label='sph')

## fig 4: s21
ax = fig.add_subplot(gs[3])
ax.plot(data_fn_cauchy[:, 4], data_fn_cauchy[:, 9] , '--r',linewidth=3, label='fenics')

## fig 5: s22
ax = fig.add_subplot(gs[4])
ax.plot(data_fn_cauchy[:, 4], data_fn_cauchy[:, 10] , '--r',linewidth=3, label='fenics')
#ax.plot(data_sph_cauchy[:, 3]-data_sph_cauchy[0, 3],  data_sph_cauchy[:, 6], 'b',linewidth=3, label='sph')
ax.plot(data_sph_cauchy_new[:, 3]-data_sph_cauchy_new[0, 3],  data_sph_cauchy_new[:, 6], 'orange',linewidth=3, label='sph')

## fig 6: s23
ax = fig.add_subplot(gs[5])
ax.plot(data_fn_cauchy[:, 4], data_fn_cauchy[:, 11] , '--r',linewidth=3, label='fenics')
#ax.plot(data_sph_cauchy[:, 3]-data_sph_cauchy[0, 3],  data_sph_cauchy[:, 10], 'b',linewidth=3, label='sph')
ax.plot(data_sph_cauchy_new[:, 3]-data_sph_cauchy_new[0, 3],  data_sph_cauchy_new[:, 10], 'orange',linewidth=3, label='sph')

## fig 7: s31
ax = fig.add_subplot(gs[6])
ax.plot(data_fn_cauchy[:, 4], data_fn_cauchy[:, 12] , '--r',linewidth=3, label='fenics')

## fig 8: s32
ax = fig.add_subplot(gs[7])
ax.plot(data_fn_cauchy[:, 4], data_fn_cauchy[:, 13] , '--r',linewidth=3, label='fenics')

## fig 9: s33
ax = fig.add_subplot(gs[8])
ax.plot(data_fn_cauchy[:, 4], data_fn_cauchy[:, 14] , '--r',linewidth=3, label='fenics')
#ax.plot(data_sph_cauchy[:, 3]-data_sph_cauchy[0, 3],  data_sph_cauchy[:, 7], 'b',linewidth=3, label='sph')
ax.plot(data_sph_cauchy_new[:, 3]-data_sph_cauchy_new[0, 3],  data_sph_cauchy_new[:, 7], 'orange',linewidth=3, label='sph')

ax.set_xlabel('Strain')
ax.set_ylabel('Stress')
ax.set_title("last")

ax.legend()   
plt.tight_layout()
fig.savefig('./Results.png')
plt.close()    
