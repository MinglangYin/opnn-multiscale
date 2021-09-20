## notice that the order of atom id may be different from timesteps.

import numpy as np
import os
import scipy.io as io

id = 21

dataset = io.loadmat("hyper_notch_100_nofrac.mat")
coord_init = dataset["coord_init"]
snapshot = dataset["snapshot"]
coord = dataset["coord"]

dump = coord_init[:, :2]
disp = coord[id] - coord_init
dump = np.concatenate((dump, disp[:, :2]))
np.savetxt("true_sol_sph.dat", dump)