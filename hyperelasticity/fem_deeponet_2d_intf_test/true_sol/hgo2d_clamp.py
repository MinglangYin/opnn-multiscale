## Author: Minglang Yin, minglang_yin@brown.edu
## Uniaxial pulling of a 3D plate
## Holzapfel-Gasser-Odgen Model
## Clamped bottom
## Impose displacement BC on the top

from dolfin import *
from mshr import *
import os
import numpy as np
import scipy.io
from scipy.interpolate import Rbf

def projection(u, pts):
    u_prj = []
    for x in pts:
        u_prj.append(u(x))
    return np.asarray(u_prj)

def data2exp_2d(data, V):
    x, y, values = data[:,0], data[:,1], data[:,2:]
    interpolant1 = Rbf(x, y, values[:, 0:1], copy=False, bounds_error=True)
    interpolant2 = Rbf(x, y, values[:, 1:2], copy=False, bounds_error=True)

    class ExpressionFromScipyFunction(UserExpression):
        def __init__(self, f, **kwargs):
            self._f1 = f[0]
            self._f2 = f[1]
            UserExpression.__init__(self, **kwargs)
        def eval(self, values, x):
            values[0] = self._f1(*x)
            values[1] = self._f2(*x)

    u = ExpressionFromScipyFunction((interpolant1, interpolant2), element=V.ufl_element())
    return u

# make new directory
path = 'Results'
os.makedirs(path, exist_ok=True)

top_disp_set = scipy.io.loadmat("../hyper_notch_100_nofrac_topdisp.mat")
top_disp = top_disp_set["disp_top"][21]
top_coord_ref = top_disp_set["coord_top_ref"][:, :2]

# Optimization options for the form compiler
parameters["form_compiler"]["cpp_optimize"] = True
ffc_options = {"optimize": True, \
               "eliminate_zeros": True, \
               "precompute_basis_const": True, \
               "precompute_ip_const": True}

# Create mesh and define function space
rectangle = Rectangle(Point(0,0),Point(1.0, 1.0))
rectangle_small = Rectangle(Point(0,0.46),Point(0.1, 0.54))
region = rectangle - rectangle_small
mesh = generate_mesh(region, 25)
mesh_coord = mesh.coordinates()

V = VectorFunctionSpace(mesh, "CG", 1)
W = TensorFunctionSpace(mesh, "CG", 1)

# Mark boundary subdomians
bot =  CompiledSubDomain("near(x[1], side) && on_boundary", side = 0.0)
top = CompiledSubDomain("near(x[1], side) && on_boundary", side = 1.0)

# Define Dirichlet boundary (x = 0 or x = 1)
bot_disp = Expression(("0.0", "0.0"), degree=1)
top_disp = data2exp_2d(np.concatenate((top_coord_ref, top_disp), axis=1), V)
bc1 = DirichletBC(V, bot_disp, bot)
bc2 = DirichletBC(V, top_disp, top)
bcs = [bc1, bc2]

# Define functions
du = TrialFunction(V)            # Incremental displacement
v  = TestFunction(V)             # Test function
u  = Function(V)                 # Displacement from previous iteration
B  = Constant((0.0, 0.0))  # Body force per unit volume

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
M1 = Constant((np.cos(alpha), np.sin(alpha)))
M2 = Constant((np.cos(-alpha), np.sin(-alpha)))

E1 = (1 - kappa/3)*(inner(C*M1, M1) - 1)
E1 = (E1+abs(E1))/2.

E2 = (1 - kappa/3)*(inner(C*M2, M2) - 1)
E2 = (E2+abs(E2))/2.

psi = (mu/2)*(Ic - 2) - mu*ln(J) + \
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
u_his = []

# Solve variational problem
# solve
solve(FF == 0, u, bcs, J=Jac, form_compiler_parameters=ffc_options)

# Cauchy stress 
S = PK2()
PK1 = project(F*S, W)   # Cauchy Stress
PK1 = projection(PK1, mesh_coord)
dump = np.concatenate((mesh_coord, projection(u, mesh_coord), PK1 ), axis= 1)

file_name = path + "/result.xdmf"
file_results = XDMFFile(file_name)
file_results.parameters["flush_output"] = True
file_results.parameters["functions_share_mesh"] = True
file_results.write(u, 0.)
#file_results.write(sigma, 0.)

np.savetxt("true_sol.dat", dump)
