import numpy as np
import matplotlib.pyplot as plt

true_out = np.loadtxt("pred_in.dat")
# true_in = np.loadtxt("")

# pred_out = np.loadtxt("")
# pred_in = np.loadtxt("")

fig = plt.figure(constrained_layout=False, figsize=(3, 3))
gs = fig.add_gridspec(1, 1)

ax = fig.add_subplot(gs[0])
h = ax.plot(true_out[:, 0], true_out[:, 1])

fig.savefig('outline.png')
plt.close()