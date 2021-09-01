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
top_exp = Expression(("t*0.0","t*0.05","t*0.0"), t = 1, degree=1)

bc1 = DirichletBC(V, bot_exp, bot)
bc2 = DirichletBC(V, top_exp, top)
bcs = [bc1, bc2]

# Define functions
du = TrialFunction(V)            # Incremental displacement
v  = TestFunction(V)             # Test function
u  = Function(V)                 # Displacement from previous iteration
u_old = Function(V) 
B  = Constant((0.0, 0.0, 0.0))  # Body force per unit volume
# T  = Constant((0.0, 0.1, 0.0))  # Traction force on the boundary

# Kinematics
d = len(u_old)
I = Identity(d)             # Identity tensor
F = I + grad(u_old)             # Deformation gradient
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
psi = (mu/2)*(Ic - 3) - mu*ln(J) + (K/2)*( (J**2-1)/2 - ln(J)) +\
      (K1/K2/2)*(exp(K2*( kappa*(Ic - 3)**2 )) - 1)

# Total potential energy
Pi = psi*dx - dot(B, u_old)*dx #- dot(T, u)*ds

F = derivative(Pi, u_old, v)      # Compute first variation of Pi (directional derivative about u in the direction of v)
J = derivative(F, u_old, du)      # Compute Jacobian of F

# Solve variational problem

for i in range(1, 3):

    file = File("disp_init.pvd")
    file << u

    top_exp.t = i
    solve(F == 0, u_old, bcs, J=J, form_compiler_parameters=ffc_options)
    u.assign(u_old)

    file_name = "displacement_" + str(i) + ".pvd"
    file = File(file_name)
    file << u
