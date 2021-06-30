"""
Author: Minglang Yin. minglang_yin@brown.edu

Coupling a FEM solver with a DeepONet for poisson equation

div(grad(u)) = -f, Omage = [0, 1]*[0, 1], f = 1
"""

import numpy as np
import matplotlib.pyplot as plt

from fenics import *
from dolfin import *
from scipy.interpolate import interp2d
from grf import *

# u_interp = interpolate(u_int, V) ## interpolate an expression onto a mesh

def data2exp(data, V):
    ## data
    x, y, values = data[:,0], data[:,1], data[:,2]
    interpolant = interp2d(x, y, values, kind='linear', copy=False, bounds_error=True)

    class ExpressionFromScipyFunction(UserExpression):
        def __init__(self, f, **kwargs):
            self._f = f
            UserExpression.__init__(self, **kwargs)
        def eval(self, values, x):
            values[:] = self._f(*x)

    u = ExpressionFromScipyFunction(interpolant, element=V.ufl_element())
    return u

def main():
    ## create mesh and define function space
    num_cases = 100
    mesh_size = 32
    mesh = UnitSquareMesh(mesh_size, mesh_size)
    V = FunctionSpace(mesh, 'P', 1)

    # x_data = np.linspace(0, 1, mesh_size+1)
    # y_data = np.linspace(0, 1, mesh_size+1)
    # xx, yy = np.meshgrid(x_data, y_data)
    # print(xx.flatten().shape)
    # exit()

    ## define the variational form
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Constant(6.0)
    a = dot(grad(u), grad(v))*dx
    L = f*v*dx

    for i in range(0, num_cases):
        print(f'generating {i}th cases.')

        grf = gaussian_random_field(alpha = 5.0, size=mesh_size+1)
        data = np.concatenate((mesh.coordinates()[:, 0:1], mesh.coordinates()[:, 1:2], grf.flatten()[:, None]), axis=1)

        ## user expression from data
        u_int = data2exp(data, V)

        ## boundary condition
        def boundary(x, on_boundary):
            return on_boundary
        bc = DirichletBC(V, u_int, boundary)

        ## solve
        u = Function(V)
        solve(a == L, u, bc)

        ## save txt
        x = V.tabulate_dof_coordinates()
        x = x.reshape((-1, mesh.geometry().dim()))
        u_dump = np.array(u.vector())[:, None]
        name = "data/xyu" + str(i) + ".dat"
        np.savetxt(name, np.concatenate((x[:,0:1], x[:,1:2], u_dump), axis=1) )
    
        ## plot 
        plot(u)
        name = 'fig/u_' + str(i) + '.png'
        plt.savefig(name)
        plt.close()

if __name__ == "__main__":
    main()