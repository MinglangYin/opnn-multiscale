import numpy as np
import matplotlib.pyplot as plt

errs = np.array([
    [1.09470592e-06, 1.18684670e-05],
    [2.5093247e-06, 2.2194765e-05],
    [3.36771867e-06, 1.32427674e-05],
    [5.23725637e-06, 1.19512598e-05],
    [5.72614611e-06, 1.37728416e-05],
    [4.16012742e-06, 1.10254831e-05],
    [1.57203675e-06, 4.44291402e-06],
    [3.58949938e-08, 5.68073987e-07]])

## figures
traction = np.linspace(0.01, 0.08, 8)
fig = plt.figure(constrained_layout=False, figsize=(5, 5))
gs = fig.add_gridspec(1, 1)
ax = fig.add_subplot(gs[0])
ax.plot(traction, errs[:, 0], color='blue', label='FEM (Disp)')
ax.plot(traction, errs[:, 1], color='red', label='DeepONet (Surf. Traction)')
# ax.axhline(y=eps, linestyle='--', color='black',  label='$\epsilon$')

ax.set_xlabel('Top Traction')
ax.set_ylabel('$L_{2}$ error')
ax.set_yscale('log')
ax.legend()
plt.savefig("convergence_f.png")
plt.close()
