from fenics import *
from ufl import nabla_div
from mshr import *

import matplotlib.pyplot as plt

def epsilon(u):
    return 0.5*(nabla_grad(u)+nabla_grad(u).T)

def sigma(u):
    return lambda_*nabla_div(u)*Identity(d) + 2*mu*epsilon(u)

## parameters
mu = 1
rho = 1
h = 0.4
lambda_ = 1.25
tol = 1e-14
g = 9.8

## mesh
square = Rectangle(Point(0,0), Point(1.0, 1.0))
circle = Circle(Point(0.5, 0.5), 0.1)
mesh = generate_mesh(square - circle, 32)

W = TensorFunctionSpace(mesh, "P", 2)
V = VectorFunctionSpace(mesh, 'P', 2)
def clamped_boundary(x, on_boundary):
    return on_boundary and x[1]<tol
bc_bot = DirichletBC(V, Constant((0, 0)), clamped_boundary)

boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundaries.set_all(0)

top = AutoSubDomain(lambda x: near(x[1], 1))
top.mark(boundaries, 1)
ds = ds(subdomain_data=boundaries)

# def inner_boundary(x, on_boundary):
#     return on_boundary and x[0]>0.3 and x[0]<0.7 and x[1]>0.3 and x[1]<0.7
# bc_in = DirichletBC(V, Constant((0, 0)), inner_boundary)

u = TrialFunction(V)
d = u.geometric_dimension()
v = TestFunction(V)
f = Constant((0, 0))
T = Constant((0, 1))
a = inner(sigma(u), epsilon(v))*dx
L = dot(f, v)*dx + dot(T, v)*ds(1) ## ds is on the boundary

u = Function(V)
solve(a == L, u, bc_bot)

sigma_u = project(sigma(u), W)

file_results = XDMFFile("elasticity_results.xdmf")
file_results.parameters["flush_output"] = True
file_results.parameters["functions_share_mesh"] = True
file_results.write(u, 0.)
file_results.write(sigma_u, 0.)