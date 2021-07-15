from fenics import *
import matplotlib.pyplot as plt
import numpy as np

# --------------------
# Functions and classes
# --------------------
def bottom(x, on_boundary):
    return (on_boundary and near(x[1], 0.0))

# Strain function
def epsilon(u):
    return sym(grad(u))

# Stress function
def sigma(u):
    return lambda_*div(u)*Identity(2) + 2*mu*epsilon(u)

# Density
rho = Constant(200.0)

# Young's modulus and Poisson's ratio
E = 0.02e9
nu = 0.0

# Lame's constants
lambda_ = E*nu/(1+nu)/(1-2*nu)
mu = E/2/(1+nu)

l_x, l_y = 5.0, 5.0  # Domain dimensions
n_x, n_y = 20, 20  # Number of elements

# Load
g_z = -2.9575e5
b_z = -10.0
g = Constant((0.0, g_z))
b = Constant((0.0, b_z))

# Model type
model = "plane_strain"
if model == "plane_stress":
    lambda_ = 2*mu*lambda_/(lambda_+2*mu)

# --------------------
# Geometry
# --------------------
mesh = Mesh("external_mesh.xml")

# Definition of Neumann condition domain
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundaries.set_all(0)

top = AutoSubDomain(lambda x: near(x[1], l_y))

top.mark(boundaries, 1)
ds = ds(subdomain_data=boundaries)

# --------------------
# Function spaces
# --------------------
V = VectorFunctionSpace(mesh, "CG", 1)
u_tr = TrialFunction(V)
u_test = TestFunction(V)

# --------------------
# Boundary conditions
# --------------------
bc = DirichletBC(V, Constant((0.0, 0.0)), bottom)

# --------------------
# Weak form
# --------------------
a = inner(sigma(u_tr), epsilon(u_test))*dx
l = rho*dot(b, u_test)*dx + inner(g, u_test)*ds(1)

# --------------------
# Solver
# --------------------
u = Function(V)
A_ass, L_ass = assemble_system(a, l, bc)

solve(A_ass, u.vector(), L_ass)

print(np.amax(u.vector()[:]))


plot(u, mode="displacement")
plt.show()