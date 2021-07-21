import numpy as np
import matplotlib.pyplot as plt
import meshio

from fenics import *
from mshr import *

## mesh
# square = Rectangle(Point(-1,-1), Point(1.0, 1.0))
# circle = Circle(Point(0.0, 0.0), 0.1)
# mesh = generate_mesh(square - circle, 25)
# mesh = refine(mesh)

# plot(mesh)
# plt.savefig("mesh.png")

# dolfin-convert plate_hole.msh plate_hole.xml

# msh = meshio.read("plate_hole.msh")
# meshio.write("plate_hole.xml",msh)

mesh = Mesh("plate_hole.xml")
plot(mesh)
plt.savefig("mesh.png")