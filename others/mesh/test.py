from fenics import *
from mshr import *
import matplotlib.pyplot as plt


# square = Rectangle(Point(-1,-1), Point(1.0, 1.0))
# circle = Circle(Point(0.0,0.0), 0.1)
# mesh = generate_mesh(square - circle, 25)

# for j in range(2):
#     mesh.init(1, 2) # Initialise facet to cell connectivity
#     markers = MeshFunction('bool', mesh, mesh.topology().dim() - 1, False)

#     class on_circle(SubDomain):
#         def inside(self, x, on_boundary):
#             return on_boundary and x[0]>-0.2 and x[0] < 0.2 and x[1]>-0.2 and x[1]<0.2

#     on_circle = on_circle()

#     on_circle.mark(markers, True)
#     mesh = refine(mesh, markers)

# plot(mesh)
# plt.savefig("mesh.png")

from fenics import *
from mshr import *
from math import sin, cos, pi

a = 1.0   # inner radius of iron cylinder
b = 1.2   # outer radius of iron cylinder
c_1 = 0.8 # radius for inner circle of copper wires
c_2 = 1.4 # radius for outer circle of copper wires
r = 0.1   # radius of copper wires
R = 5.0   # radius of domain
n = 10    # number of windings

##: Use 'domain' instead of 'geometry' in other examples

# Define geometry for background
domain = Circle(Point(0, 0), R)

# Define geometry for iron cylinder
cylinder = Circle(Point(0, 0), b) - Circle(Point(0, 0), a)

# Define geometry for wires (N = North (up), S = South (down))
angles_N = [i*2*pi/n for i in range(n)]
angles_S = [(i + 0.5)*2*pi/n for i in range(n)]
wires_N = [Circle(Point(c_1*cos(v), c_1*sin(v)), r) for v in angles_N]
wires_S = [Circle(Point(c_2*cos(v), c_2*sin(v)), r) for v in angles_S]

# Set subdomain for iron cylinder
domain.set_subdomain(1, cylinder)

# Set subdomains for wires
for (i, wire) in enumerate(wires_N):
    domain.set_subdomain(2 + i, wire)
for (i, wire) in enumerate(wires_S):
    domain.set_subdomain(2 + n + i, wire)

# Create mesh
mesh = generate_mesh(domain, 32)
plot(mesh)
plt.savefig("mesh.png")