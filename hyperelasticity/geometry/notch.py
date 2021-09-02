import numpy as np
import os
import glob
import sys

def main():
    x_lim = [0, 1]
    y_lim = [0, 1]
    x = np.linspace(x_lim[0], x_lim[1], 101)
    y = np.linspace(y_lim[0], y_lim[1], 101)
    xx, yy = np.meshgrid(x, y)
    zz = np.ones_like(xx)*0

    coords = np.concatenate((xx.flatten()[:, None], yy.flatten()[:, None], 
                            zz.flatten()[:, None] ), axis=1)
    
    new_coords = []
    atom_id = 0
    atom_type = 1
    molecule = 1
    volume = 3.08893
    mass = 6.17786e-6
    kradius = 1.89656
    cradius = 0.497594
    prefix = [atom_type, molecule, volume, mass, kradius, cradius]
    for coord in coords:
        if coord[0] <= 0.1 and coord[1] == 0.5: ## where the notch locates
            continue
        else:
            atom_id = atom_id + 1

            ## atom-ID atom-type molecule volume mass kradius/kernel_radius cradius/contact_radius x0 y0 z0 x y z
            print(coord.shape)
            tmp = np.concatenate((atom_id,  prefix, coord, coord), axis=0)
            print(tmp)
            new_coords.append(tmp)
    exit()

    fname = "notch.data"
    with open(fname,'w') as f:
        f.write("LAMMPS\n")
        f.write("%d atoms\n" %(atom_id))
        f.write("1 atom types\n")
        f.write("%f     %f      xlo xhi\n" %(x_lim[0], x_lim[1]))
        f.write("%f     %f      ylo yhi\n" %(x_lim[0], x_lim[1]))
        f.write("0     0.01      zlo zhi\n")
        f.write("\n")
        f.write("Atoms\n")

        for coord in new_coords:
            f.write(str(coord))
            f.write("\n")

        f.write("\n")
        f.write("Velocities\n")
        for id, _ in enumerate(coords):
            f.write("%d 0.0 0.0 0.0\n" %(id))
        
        f.write("\n")
        f.close()

if __name__ == "__main__":
    main()