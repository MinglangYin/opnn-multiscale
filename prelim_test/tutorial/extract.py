from dolfin import *
import matplotlib.pyplot as plt
import numpy as np

mesh = UnitSquareMesh(100, 100, "crossed")
V = FunctionSpace(mesh, "CG", 2)
u = interpolate(Expression("x[0]*x[0] + x[1]*x[1]", degree=1), V)

print(np.array(u.vector()).shape )

class half(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] > 0.5

cell_f = MeshFunction("size_t", mesh, mesh.topology().dim(), 0)
cell_f.set_all(0)
half = half()

# mark subdomains
for cell in cells(mesh):
    if half.inside(cell.midpoint(), True):
        cell_f[cell.index()] = 1

def extract_values(u, cell_function, subdomain_id, V):
    dofmap = V.dofmap()
    mesh = V.mesh()
    for cell in cells(mesh):
        # Preserve only the dofs in cells marked as subdomain_id
        if cell_function[cell.index()] != subdomain_id:
            dofs = dofmap.cell_dofs(cell.index())
            for dof in dofs:
                u.vector()[dof] = 0.0
    return u


u_left  = Function(V); u_left.vector()[:]  = u.vector()
u_right = Function(V); u_right.vector()[:] = u.vector()
u_left  = extract_values(u_left, cell_f, 0, V)
u_right = extract_values(u_right, cell_f, 1, V)

plot(u_left, title="left")
# plot(u_right, title="right")
plt.savefig('test.png')