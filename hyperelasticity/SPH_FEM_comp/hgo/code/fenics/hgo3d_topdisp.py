## Author: Minglang Yin, minglang_yin@brown.edu
## Uniaxial pulling of a 3D plate
## Holzapfel-Gasser-Odgen Model
## Free to move in x direction
## Impose displacement BC on the top, validated

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

# Mark boundary subdomians
topdisp = Expression(("t*0.025"), t = 1, degree=1)
bot =  CompiledSubDomain("near(x[1], side) && on_boundary", side = 0.0)
top =  CompiledSubDomain("near(x[1], side) && on_boundary", side = 1.0)
back =  CompiledSubDomain("near(x[2], side) && on_boundary", side = 0.0)
bc1 = DirichletBC(V.sub(1), Constant((0.0)), bot) # v(*, 0) = 0
bc2 = DirichletBC(V.sub(1), topdisp, top) 
bc3 = DirichletBC(V.sub(2), Constant((0.0)), back) # v(*, 0) = 0
bcs = [bc1, bc2, bc3]

# Define functions
du = TrialFunction(V)            # Incremental displacement
v  = TestFunction(V)             # Test function
u  = Function(V)                 # Displacement from previous iteration
B  = Constant((0.0, 0.0, 0.0))  # Body force per unit volume

# Kinematics
d = len(u)
I = Identity(d)             # Identity tensor
F = I + grad(u)             # Deformation gradient
C = F.T*F                   # Right Cauchy-Green tensor
b = F*F.T                   # Left Cauchy-Green tensor

# Invariants of deformation tensors
Ic = tr(C)
J  = det(F)

# Elasticity parameters
E, nu = 1.0, 0.3
mu, lmbda = E/(2*(1 + nu)), E*nu/((1 + nu)*(1 - 2*nu))
K = E/(3*(1-2*nu)) # kappa

print(f'mu: {mu}, lmbda: {lmbda}, K: {K}')

# Stored strain energy density (compressible neo-Hookean model)
K1, K2 = 0.1, 1.5
alpha = np.pi/2      
kappa = 0
M1 = Constant((np.cos(alpha), np.sin(alpha), 0))
M2 = Constant((np.cos(-alpha), np.sin(-alpha), 0))

E1 = (1 - kappa/3)*(inner(C*M1, M1) - 1)
E1 = (E1+abs(E1))/2.

E2 = (1 - kappa/3)*(inner(C*M2, M2) - 1)
E2 = (E2+abs(E2))/2.

psi = (mu/2)*(Ic - 3) - mu*ln(J) + \
        (K1/K2/2)*(exp(K2*E1**2) - 1) + (K1/K2/2)*(exp(K2*E2**2) - 1) + \
        (K/2)*( (J**2 - 1)/2 - ln(J) )

# Total potential energy
Pi = psi*dx - dot(B, u)*dx
FF = derivative(Pi, u, v)      # Compute first variation of Pi (directional derivative about u in the direction of v)
Jac = derivative(FF, u, du)      # Compute Jacobian of F

# PK2 stress
def PK2():
    return mu*(I - inv(C)) + 2*K1*E1*(exp(K2*E1**2) )*outer(M1, M1) + 2*K1*E2*(exp(K2*E2**2) )*outer(M2, M2) + \
       K/2*inv(C)*(J**2 - 1)

# 
x_plot = np.array([0.5, 1.0, 0.0])
u_his = []

# Solve variational problem
for i in range(0, 15):
    # solve
    topdisp.t = i
    solve(FF == 0, u, bcs, J=Jac, form_compiler_parameters=ffc_options)

    # Cauchy stress 
    S = PK2()
    sigma = project(F*S*F.T/J, W)   # Cauchy Stress
    sigma_y = sigma(x_plot)[4:5]
    print(sigma_y)
    dump = np.concatenate((x_plot, u(x_plot), sigma_y ), axis= 0)
    u_his.append(dump)

    file_name = path + "/result_" + str(i) + ".xdmf"
    file_results = XDMFFile(file_name)
    file_results.parameters["flush_output"] = True
    file_results.parameters["functions_share_mesh"] = True
    file_results.write(u, 0.)
    file_results.write(sigma, 0.)

np.savetxt("stress_strain.dat", np.array(u_his))