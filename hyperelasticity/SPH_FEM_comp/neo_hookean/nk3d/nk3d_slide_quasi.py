from dolfin import *
from mshr import *
import os
import numpy as np

# make new directory
path = 'Results'
os.makedirs(path, exist_ok=True)

# Optimization options for the form compiler
parameters["form_compiler"]["cpp_optimize"] = True
ffc_options = {"optimize": True, \
               "eliminate_zeros": True, \
               "precompute_basis_const": True, \
               "precompute_ip_const": True}

# Create mesh and define function space
mesh = BoxMesh(Point(0, 0, 0), Point(1.0, 1.0, 0.1), 25, 25, 2)
V = VectorFunctionSpace(mesh, "CG", 1)
W = TensorFunctionSpace(mesh, "CG", 1)

boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundaries.set_all(0)
top = AutoSubDomain(lambda x: x[1] > 1.0-1e-5)
top.mark(boundaries, 1)
ds = ds(subdomain_data=boundaries)

# Mark boundary subdomians
bot =  CompiledSubDomain("near(x[1], side) && on_boundary", side = 0.0)
back =  CompiledSubDomain("near(x[2], side) && on_boundary", side = 0.0)
bc1 = DirichletBC(V.sub(1), Constant((0.0)), bot) # v(*, 0) = 0
bc2 = DirichletBC(V.sub(2), Constant((0.0)), back) # v(*, 0) = 0
bcs = [bc1, bc2]

# Define functions
du = TrialFunction(V)            # Incremental displacement
v  = TestFunction(V)             # Test function
u  = Function(V)                 # Displacement from previous iteration
B  = Constant((0.0, 0.0, 0.0))  # Body force per unit volume
t_top  = Expression(("t*0.0","t*0.15","t*0.0"), t = 0, degree=1)  # PK1 Traction force on the boundary

# Kinematics
d = len(u)
I = Identity(d)             # Identity tensor
F = I + grad(u)             # Deformation gradient
C = F.T*F                   # Right Cauchy-Green tensor
b = F*F.T                   # Left Cauchy-Green tensor
C = variable(C)
# EL = 0.5*(C - I)
# EL = variable(EL)

# Invariants of deformation tensors
Ic = tr(C)
J  = det(F)

# Elasticity parameters
E, nu = 1.0, 0.3
mu, lmbda = Constant(E/(2*(1 + nu))), Constant(E*nu/((1 + nu)*(1 - 2*nu)))
K = E/(3*(1-2*nu))

# Stored strain energy density (compressible neo-Hookean model)
psi = (mu/2)*(Ic - 3 - 2*ln(J)) + (lmbda/2)*ln(J)**2 # +\
      #(K1/K2/2)*(exp(K2*( kappa*(Ic - 3)**2 )) - 1)

# Total potential energy
Pi = psi*dx - dot(B, u)*dx - dot(t_top, u)*ds(1)
FF = derivative(Pi, u, v)      # Compute first variation of Pi (directional derivative about u in the direction of v)
Jac = derivative(FF, u, du)      # Compute Jacobian of F

## PK2 stress
def PK2():
    return mu*(I - inv(C)) + lmbda*ln(J)*inv(C)

# 
x_plot = np.array([0.5, 1.0, 0.0])
u_his = []

# Solve variational problem
for i in range(0, 11):
    # solve
    t_top.t = i
    solve(FF == 0, u, bcs, J=Jac, form_compiler_parameters=ffc_options)

    # Approach 1: Cauchy stress
    def sigma(u):
        return mu/J*(b - I) + lmbda/J*ln(J)*I
    sigma = project(sigma(u), W)
    sigma_y = sigma(x_plot)
    print(sigma_y)

    ## Approach 2: PK2 -> Cauchy
    S = PK2()
    sigma = project(F*S*F.T/J, W)
    sigma_y = sigma(x_plot)
    print(sigma_y)

    ## another way to compute sigma
    ## Fail. Because the energy is not a function of C explicitly. psi=f(C) 
    # S = 2*diff(psi, C) # PK-2
    # sigma = project(S, W)
    # sigma_y = sigma(x_plot)
    # print(sigma_y)

    # save
    dump = np.concatenate((x_plot, u(x_plot), sigma_y), axis= 0)
    u_his.append(dump)

    file_name = path + "/result_" + str(i) + ".xdmf"
    file_results = XDMFFile(file_name)
    file_results.parameters["flush_output"] = True
    file_results.parameters["functions_share_mesh"] = True
    file_results.write(u, 0.)
    file_results.write(sigma, 0.)

np.savetxt("stress_strain.dat", np.array(u_his))