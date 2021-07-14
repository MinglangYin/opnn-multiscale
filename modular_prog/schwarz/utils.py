import numpy as np
import torch
import matplotlib.pyplot as plt

from dolfin import *
from mshr import *
from scipy.interpolate import interp2d

def data2exp(data, V):
    x, y, values = data[:,0], data[:,1], data[:,2]
    interpolant = interp2d(x, y, values, kind='cubic', copy=False, bounds_error=True)

    class ExpressionFromScipyFunction(UserExpression):
        def __init__(self, f, **kwargs):
            self._f = f
            UserExpression.__init__(self, **kwargs)
        def eval(self, values, x):
            values[:] = self._f(*x)

    u = ExpressionFromScipyFunction(interpolant, element=V.ufl_element())
    return u

def to_numpy(input):
    if isinstance(input, torch.Tensor):
        return input.detach().cpu().numpy()
    elif isinstance(input, np.ndarray):
        return input
    else:
        raise TypeError('Unknown type of input, expected torch.Tensor or '\
            'np.ndarray, but got {}'.format(type(input)))

## temporary method for generating square meshes
def gen_mesh():
    dim = 17
    gemma0_x, gemma0_y = 0.0, 1.0
    gemma1_x, gemma1_y = 0.4, 0.6
    gemma2_x, gemma2_y = 0.3, 0.7

    ## mesh 1
    box = Rectangle(Point(gemma0_x, gemma0_x), Point(gemma0_y, gemma0_y))
    box_in = Rectangle(Point(gemma1_x, gemma1_x), Point(gemma1_y, gemma1_y))
    omega_s = box - box_in
    mesh_s = generate_mesh(omega_s, dim)

    ## mesh 2
    omega_p = Rectangle(Point(gemma2_x, gemma2_x), Point(gemma2_y, gemma2_y))
    mesh_p = generate_mesh(omega_p, dim)

    def get_bound_coord(archor_p1, archor_p2, dim):
        x_min, y_min = archor_p1[0], archor_p1[1]
        x_max, y_max = archor_p2[0], archor_p2[1]

        vline = np.linspace(x_min, x_max, dim)[:, None]
        hline = np.linspace(y_min, y_max, dim)[:, None]
        ones = np.ones((dim, 1))

        top = np.concatenate( (vline, ones*y_max), axis=1 )
        bot = np.concatenate( (vline, ones*y_min), axis=1 )
        left = np.concatenate( (ones*x_min, hline), axis=1 )
        right = np.concatenate( (ones*x_max, hline), axis=1 )

        # interf_coord = np.concatenate((top, bot, left, right), axis=0)
        return top, bot, left, right

    top_o, bot_o, left_o, right_o = get_bound_coord([0, 0], [1.0, 1.0], dim)
    top_g1, bot_g1, left_g1, right_g1 = get_bound_coord([gemma1_x, gemma1_x], [gemma1_y, gemma1_y], dim)
    top_g2, bot_g2, left_g2, right_g2 = get_bound_coord([gemma2_x, gemma2_x], [gemma2_y, gemma2_y], dim)

    x_g1 = np.concatenate((top_g1, bot_g1, left_g1, right_g1), axis=0)
    x_g2 = np.concatenate((top_g2, bot_g2, left_g2, right_g2), axis=0)
    x_go = np.concatenate((top_o, bot_o, left_o, right_o), axis=0)

    x_s = mesh_s.coordinates()
    x_p = mesh_p.coordinates()

    return mesh_s, mesh_p, x_g1, x_g2, x_go, x_s, x_p

# def load_mesh(**kwargs):

#     gemma1_x, gemma1_y = 0.4, 0.6
#     gemma2_x, gemma2_y = 0.3, 0.7

#     box = Rectangle(Point(0,0), Point(1.0, 1.0))
#     box_in = Rectangle(Point(gemma1_x, gemma1_x), Point(gemma1_y, gemma1_y))
#     omega1 = box - box_in
#     mesh1 = generate_mesh(omega1, 16)

#     omega2 = Rectangle(Point(gemma2_x, gemma2_x), Point(gemma2_y, gemma2_y))
#     mesh2 = generate_mesh(omega2, 16)

#     return mesh1, mesh2, xg1, xg2
