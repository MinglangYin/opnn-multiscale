import torch
import numpy as np

from fenics import *
from mshr import *
from scipy.interpolate import Rbf

def to_numpy(input):
    if isinstance(input, torch.Tensor):
        return input.detach().cpu().numpy()
    elif isinstance(input, np.ndarray):
        return input
    else:
        raise TypeError('Unknown type of input, expected torch.Tensor or '\
            'np.ndarray, but got {}'.format(type(input)))

def projection(u, pts):
    u_prj = []
    for x in pts:
        u_prj.append(u(x))
    return np.asarray(u_prj)

def data2exp_2d(data, V):
    x, y, values = data[:,0], data[:,1], data[:,2:]
    interpolant1 = Rbf(x, y, values[:, 0:1], copy=False, bounds_error=True)
    interpolant2 = Rbf(x, y, values[:, 1:2], copy=False, bounds_error=True)

    class ExpressionFromScipyFunction(UserExpression):
        def __init__(self, f, **kwargs):
            self._f1 = f[0]
            self._f2 = f[1]
            UserExpression.__init__(self, **kwargs)
        def eval(self, values, x):
            values[0] = self._f1(*x)
            values[1] = self._f2(*x)

    u = ExpressionFromScipyFunction((interpolant1, interpolant2), element=V.ufl_element())
    return u

def find_outside(x, center_coord, r0):
    list = []
    x0, x1 = center_coord[0], center_coord[1]
    for pt in x:
        if ( (pt[0]-x0)**2+(pt[1]-x1)**2) > r0**2:
            list.append(pt)
    return np.array(list)

def circ_noraml(x, center):
    norm = np.linalg.norm(x-center, ord=2, axis=1)[:, None]
    return (x-center)/norm