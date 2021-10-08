import numpy as np
import os
from sklearn import gaussian_process as gp
from pathos.pools import ProcessPool
from scipy import interpolate
import matplotlib.pyplot as plt
from sklearn import preprocessing

os.makedirs("fig", exist_ok=True)

processes = 4
class GRF(object):
    def __init__(self, T, kernel="RBF", length_scale=1, N=1000, interp="cubic"):
        self.N = N
        self.interp = interp
        self.x = np.linspace(0, T, num=N)[:, None]
        if kernel == "RBF":
            K = gp.kernels.RBF(length_scale=length_scale)
        elif kernel == "AE":
            K = gp.kernels.Matern(length_scale=length_scale, nu=0.5)
        K = K(self.x)
        self.L = np.linalg.cholesky(K + 1e-13 * np.eye(self.N))

    def random(self, n):
        """Generate `n` random feature vectors.
        """
        u = np.random.randn(self.N, n)
        return np.dot(self.L, u).T

    def eval_u_one(self, y, x):
        """Compute the function value at `x` for the feature `y`.
        """
        if self.interp == "linear":
            return np.interp(x, np.ravel(self.x), y)
        f = interpolate.interp1d(
            np.ravel(self.x), y, kind=self.interp, copy=False, assume_sorted=True
        )
        return f(x)

    def eval_u(self, ys, sensors):
        """For a list of functions represented by `ys`,
        compute a list of a list of function values at a list `sensors`.
        """
        if self.interp == "linear":
            return np.vstack([np.interp(sensors, np.ravel(self.x), y).T for y in ys])
        p = ProcessPool(nodes=processes)
        res = p.map(
            lambda y: interpolate.interp1d(
                np.ravel(self.x), y, kind=self.interp, copy=False, assume_sorted=True
            )(sensors).T,
            ys,
        )
        return np.vstack(list(res))


space = GRF(1, length_scale=0.3, N=100, interp="cubic")
m = 100
features = space.random(2)
x = np.linspace(0, 1, num=m)[:, None]
num_sample = 100
data = np.zeros((m, num_sample+1))
data[:, 0:1] = x

d_scal = 0.001

for id in range(0, num_sample):
    features = space.random(1) 
    sensor_values = space.eval_u(features, x)[0]
    normalized =  (sensor_values-min(sensor_values))/(max(sensor_values)-min(sensor_values))
    data[:, id+1] = normalized*d_scal

    ## curve fitting
    

    ## figure
    fig = plt.figure(constrained_layout=False, figsize=(5, 5))
    gs = fig.add_gridspec(1, 1)
    ax = fig.add_subplot(gs[0])
    h = ax.plot(normalized, linewidth=3)
    name = "./fig/disp_" + str(id) + ".png"
    fig.savefig(name)
    plt.close()

np.savetxt("rand_disp.dat", data)