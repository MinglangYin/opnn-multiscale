import numpy as np
import matplotlib.pyplot as plt

from dolfin import *
from mshr import *

def main():
    beta = 8
    R0 = 0.6
    p = Expression('4*exp(-beta*beta*(x[0]*x[0]+(x[1]-R0)*(x[1]-R0)))',degree=1, beta=beta, R0=R0)

    ## Create mesh and define function space
    domain = Circle(Point(0, 0), 1)
    mesh = generate_mesh(domain, 64)

    plot(mesh)
    plt.savefig("circle_mesh.png")
    exit()

    V = FunctionSpace(mesh, 'P', 1)
    w = TrialFunction(V)
    v = TestFunction(V)
    a = dot(grad(w), grad(v))*dx
    L = p*v*dx
    
    ## Boundary condition
    def boundary(x, on_boundary):
        return on_boundary

    u_D = 0
    bc = DirichletBC(V, u_D, boundary)

    ## Compute solution
    w = Function(V) ## function object representing the solution
    solve(a == L, w, bc)

    ## interpolation
    p = interpolate(p, V)

    ## Save vtk
    vtkfile_w = File('poisson_membrane/deflection.pvd')
    vtkfile_w << w
    vtkfile_p = File('poisson_membrane/load.pvd')
    vtkfile_p << p

    ## plotting
    tol = 0.001
    y = np.linspace(-1+tol, 1-tol, 101)
    points = [(0, y_) for y_ in y]
    w_line = np.array([w(point) for point in points])
    p_line = np.array([p(point) for point in points])
    plt.plot(y, 50*w_line, 'k', linewidth=2)  # magnify w
    plt.plot(y, p_line, '--b', linewidth=2)
    plt.grid(True)
    plt.xlabel('$y$')
    plt.legend(['Deflection ($\\times 50$)', 'Load'], loc='upper left')
    plt.savefig('poisson_membrane/curves.pdf')
    plt.savefig('poisson_membrane/curves.png')

if __name__ == "__main__":
    main()