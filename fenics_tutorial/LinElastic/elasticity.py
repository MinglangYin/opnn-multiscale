from fenics import *
from ufl import nabla_div

L = 1
W = 0.2
mu = 1
rho = 1
delta = W/L
gamma = 0.4*delta**2
beta = 1.25
lambda_ = beta
g = gamma
tol = 1e-14

mesh = BoxMesh(Point(0,0,0), Point(L, W, W), 10, 3, 3)
V = VectorFunctionSpace(mesh, 'P', 1)

def clamped_boundary(x, on_boundary):
    return on_boundary and x[0]<tol

bc = DirichletBC(V, Constant((0, 0, 0)), clamped_boundary)

def epsilon(u):
    return 0.5*(nabla_grad(u)+nabla_grad(u).T)

def sigma(u):
    return lambda_*nabla_div(u)*Identity(d) + 2*mu*epsilon(u)

u = TrialFunction(V)
d = u.geometric_dimension()
v = TestFunction(V)
f = Constant((0, 0, -rho*g))
T = Constant((0, 0, 0))
a = inner(sigma(u), epsilon(v))*dx
L = dot(f, v)*dx + dot(T, v)*ds ## ds is on the boundary

u = Function(V)
solve(a == L, u, bc)
# u = project(u, V)

# u1, u2, u3 = u.split()

file = File("displacement.pvd")
file << u

