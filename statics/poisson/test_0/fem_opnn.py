"""
Author: Minglang Yin. minglang_yin@brown.edu

Coupling a FEM solver with a DeepONet for poisson equation

div(grad(u)) = -f, Omage = [0, 1]*[0, 1], f = 1
"""

import numpy as np
import matplotlib.pyplot as plt

from fenics import *

def get_inner(mesh, u, V):
    # define inner boundary condition
    class inner(SubDomain):
        def inside(self, x, on_boundary):
            return x[0] > 0.25-DOLFIN_EPS and x[0] < 0.75+DOLFIN_EPS \
                and x[1] > 0.25-DOLFIN_EPS and x[1] < 0.75+DOLFIN_EPS

    cell_in = MeshFunction("size_t", mesh, mesh.topology().dim(), 0)
    cell_in.set_all(0)
    inner_domain = inner()

    for cell in cells(mesh):
        if inner_domain.inside(cell.midpoint(), True):
            cell_in[cell.index()] = 1

    def extract_values(u, cell_function, subdomain_id, V):
        dofmap = V.dofmap()
        mesh = V.mesh()
        u_in = []
        x_in = []
        y_in = []
        i = 0
        for cell in cells(mesh):
            # Preserve only the dofs in cells marked as subdomain_id
            if cell_function[cell.index()] != subdomain_id:
                dofs = dofmap.cell_dofs(cell.index())
                for dof in dofs:
                    u.vector()[dof] = 0.0
            # else:
                # print(cell.get_vertex_coordinates())
                # u_in.append()
                # x_in.append()
                # y_in.append()
        return u

    # u_in = Function(V)
    # u.vector()[:] = u.vector()
    u_in = extract_values(u, cell_in, 1, V)
    plot(u_in, title="in")
    plt.savefig('test.png')

def main():
    ## create mesh and define function space
    mesh = UnitSquareMesh(64, 64)
    V = FunctionSpace(mesh, 'P', 1)

    ## define boundary condition
    u_D = Expression('1 + x[0]*x[0] + 2*x[1]*x[1]', degree=2)

    def boundary(x, on_boundary):
        return on_boundary

    # def boundary_inner(x, on_boundary):
        # return 'x[0]>0.25' and 'x[0]<0.75' and 'x[1]>0.25' and 'x[1]<0.75'

    bc = DirichletBC(V, u_D, boundary)

    ## define the variational form
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Constant(-6.0)
    a = dot(grad(u), grad(v))*dx
    L = f*v*dx

    ## solve
    u = Function(V)
    solve(a == L, u, bc)

    ## get_inside
    # get_inner(mesh, u, V)
    # exit()

    ## save txt
    x = V.tabulate_dof_coordinates()
    x = x.reshape((-1, mesh.geometry().dim()))
    u_dump = np.array(u.vector())[:, None]
    np.savetxt("xyu.txt", np.concatenate((x[:,0:1], x[:,1:2], u_dump), axis=1) )
    
if __name__ == "__main__":
    main()