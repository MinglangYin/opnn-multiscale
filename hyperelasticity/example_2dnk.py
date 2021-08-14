from dolfin import *
from mshr import *

# Optimization options for the form compiler
parameters["form_compiler"]["cpp_optimize"] = True
ffc_options = {"optimize": True, \
               "eliminate_zeros": True, \
               "precompute_basis_const": True, \
               "precompute_ip_const": True}

# Create mesh and define function space
mesh = BoxMesh(Point(0,0,0), Point(1.0, 1.0, 0.05), 25, 25, 1)
V = VectorFunctionSpace(mesh, "Lagrange", 1)

# Mark boundary subdomians
bot =  CompiledSubDomain("near(x[1], side) && on_boundary", side = 0.0)
top = CompiledSubDomain("near(x[1], side) && on_boundary", side = 1.0)

# Define Dirichlet boundary (x = 0 or x = 1)
bot_exp = Expression(("0.0", "0.0", "0.0"), degree=1)
# top_exp = Expression(("scale*0.0","scale*1.0","scale*0.0"), scale = 0.5, degree=1)

bc1 = DirichletBC(V, bot_exp, bot)
# bc2 = DirichletBC(V, top_exp, top)
# bcs = [bc1, bc2]
bcs = [bc1]

# Define functions
du = TrialFunction(V)            # Incremental displacement
v  = TestFunction(V)             # Test function
u  = Function(V)                 # Displacement from previous iteration
B  = Constant((0.0, 0.0, 0.0))  # Body force per unit volume
T  = Constant((0.0, 0.1, 0.0))  # Traction force on the boundary

# Kinematics
d = len(u)
I = Identity(d)             # Identity tensor
# print(V.ufl_cell().d)
# I = Identity(V.cell().d)    # Identity tensor
F = I + grad(u)             # Deformation gradient
C = F.T*F                   # Right Cauchy-Green tensor

# Invariants of deformation tensors
Ic = tr(C)
J  = det(F)

# Elasticity parameters
E, nu = 10.0, 0.3
mu, lmbda = Constant(E/(2*(1 + nu))), Constant(E*nu/((1 + nu)*(1 - 2*nu)))
K = E/(3*(1-2*nu))
K1 = 1
K2 = 1
kappa = 1/3

# Stored strain energy density (compressible neo-Hookean model)
#psi = (mu/2)*(Ic - 3) - mu*ln(J) + (lmbda/2)*(ln(J))**2
psi = (mu/2)*(Ic - 3) - mu*ln(J) + (K/2)*( (J**2-1)/2 - ln(J)) +\
      (K1/K2/2)*(exp(K2*( kappa*(Ic - 3)**2 )) - 1)

# Total potential energy
Pi = psi*dx - dot(B, u)*dx - dot(T, u)*ds

# Compute first variation of Pi (directional derivative about u in the direction of v)
F = derivative(Pi, u, v)

# Compute Jacobian of F
J = derivative(F, u, du)

# Solve variational problem
solve(F == 0, u, bcs, J=J,
      form_compiler_parameters=ffc_options)

# Save solution in VTK format
file = File("displacement.pvd")
file << u
