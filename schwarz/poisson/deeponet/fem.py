"""
Author: Minglang Yin. minglang_yin@brown.edu

Coupling a FEM solver with a DeepONet for poisson equation

div(grad(u)) = -f, Omage = [0, 1]*[0, 1], f = 1
"""

import numpy as np
import matplotlib.pyplot as plt

from fenics import *
from scipy.interpolate import interp2d
# from grf import *

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
    num_cases = 1000 # 100 
    mesh_size = 40
    mesh = UnitSquareMesh(mesh_size, mesh_size)
    V = FunctionSpace(mesh, 'P', 1)

    ## define the variational form
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Constant(-6.0)
    a = dot(grad(u), grad(v))*dx
    L = f*v*dx

    for i in range(0, num_cases):
        print(f'generating {i}th cases.')

        aa = 2*np.random.random_sample()
        bb = 3*np.random.random_sample()
        cc = 4*np.random.random_sample()

        print(aa, bb, cc)

        u_D = Expression('a + b*x[0]*x[0] + c*x[1]*x[1]', a=aa, b=bb, c=cc, degree=2)

        ## boundary condition
        def boundary(x, on_boundary):
            return on_boundary
        bc = DirichletBC(V, u_D, boundary)

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