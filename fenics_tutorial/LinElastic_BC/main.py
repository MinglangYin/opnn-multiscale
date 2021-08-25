"""
Minglang Yin, minglang_yin@brown.edu
"""
import numpy as np
import matplotlib.pyplot as plt

from fenics import *
from ufl import nabla_div
from mshr import *

def epsilon(u):
    return 0.5*(nabla_grad(u)+nabla_grad(u).T)

def sigma(u):
    return lambda_*nabla_div(u)*Identity(d) + 2*mu*epsilon(u)

## parameters
mu = 1/(2*1.3)   # E/(2*(1+nu))
lambda_ = 0.3/(1.3*0.4)   # E*nu/((1+nu)*(1-2*nu))
eps = 1e-6

########## FEM ##########
## mesh
square = Rectangle(Point(-1.0, -1.0), Point(1.0, 1.0))
mesh = generate_mesh(square, 32)
mesh_coord = mesh.coordinates()

## Space and Boundary
W = TensorFunctionSpace(mesh, "P", 2)
V = VectorFunctionSpace(mesh, "P", 2)

boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundaries.set_all(0)
top = AutoSubDomain(lambda x: x[1] > 1.0-1e-5)
top.mark(boundaries, 1)
mesh_intf = AutoSubDomain(lambda x: sqrt(x[0]*x[0]+x[1]*x[1]) < 0.3+1e-5)
mesh_intf.mark(boundaries, 2)
ds_fem = ds(subdomain_data=boundaries)

def clamped_boundary(x, on_boundary):
    return on_boundary and x[1] < -1 + eps
# bc_bot = DirichletBC(V, Constant((0, 0)), clamped_boundary)
bc_bot = DirichletBC(V.sub(1), Constant((0.0)), clamped_boundary) ## fix y displace

## Initialization
u = TrialFunction(V)
d = u.geometric_dimension()
v = TestFunction(V)
f = Constant((0, 0))
T = Constant((0, 0.4))
a = inner(sigma(u), epsilon(v))*dx
L = dot(f, v)*dx + dot(T, v)*ds_fem(1) ## ds is on the boundary

u = Function(V)
solve(a == L, u, bc_bot)

file = File("displacement.pvd")
file << u