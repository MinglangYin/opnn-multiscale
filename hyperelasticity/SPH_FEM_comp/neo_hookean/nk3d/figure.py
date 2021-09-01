import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("stress_strain.dat")

fig = plt.figure(constrained_layout=False, figsize=(5, 5))
gs = fig.add_gridspec(1, 1)
ax = fig.add_subplot(gs[0])
h = ax.plot(data[:, 4], data[:, -1], label="True")
fig.savefig('str_disp.png')
plt.close()