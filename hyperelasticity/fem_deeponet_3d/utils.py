import torch
import numpy as np
import matplotlib.pyplot as plt

from fenics import *
from ufl import nabla_div
from mshr import *
from scipy.interpolate import interp2d
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

def data2exp_3d(data, V):
    x, y, z, values = data[:,0], data[:,1], data[:,2], data[:,3:]
    interpolant1 = Rbf(x, y, z, values[:, 0:1], copy=False, bounds_error=True)
    interpolant2 = Rbf(x, y, z, values[:, 1:2], copy=False, bounds_error=True)
    interpolant3 = Rbf(x, y, z, values[:, 2:3], copy=False, bounds_error=True)

    class ExpressionFromScipyFunction(UserExpression):
        def __init__(self, f, **kwargs):
            self._f1 = f[0]
            self._f2 = f[1]
            self._f3 = f[2]
            UserExpression.__init__(self, **kwargs)
        def eval(self, values, x):
            values[0] = self._f1(*x)
            values[1] = self._f2(*x)
            values[2] = self._f3(*x)

    u = ExpressionFromScipyFunction((interpolant1, interpolant2, interpolant3), element=V.ufl_element())
    return u
