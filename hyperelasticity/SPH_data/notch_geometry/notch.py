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
    atom_type = 1
    molecule = 1
    rho = 1
    l0 = 0.01
    volume = l0*2*np.pi
    mass = rho*volume
    kradius = l0*4.01
    cradius = l0*0.1
    num_atom = 0
    for coord in coords:
        if coord[0] <= 0.1 and coord[1] >= 0.47 and coord[1] <= 0.53: ## where the notch locates
            continue
        else:
            num_atom = num_atom + 1
            new_coords.append(coord)

    fname = "notch.data"
    with open(fname,'w') as f:
        f.write("LAMMPS\n")
        f.write("%d atoms\n" %(num_atom))
        f.write("1 atom types\n")
        f.write("%f     %f      xlo xhi\n" %(x_lim[0]-0.1, x_lim[1]+0.1))
        f.write("%f     %f      ylo yhi\n" %(x_lim[0]-0.1, x_lim[1]+0.1))
        f.write("0     0.01      zlo zhi\n")
        f.write("\n")
        f.write("Atoms\n")
        f.write("\n")

        atom_id = 0
        for coord in new_coords:
            atom_id = atom_id + 1
            ## atom-ID atom-type molecule volume mass kradius/kernel_radius cradius/contact_radius x0 y0 z0 x y z
            f.write("%d %d %d %f %f %f %f %f %f %f %f %f %f\n" %(atom_id, atom_type, 
                molecule, volume, mass, kradius, cradius, coord[0], coord[1], 
                coord[2], coord[0], coord[1], coord[2]))

        f.write("\n")
        f.write("Velocities\n")
        f.write("\n")
        for id in range(1, num_atom+1):
            f.write("%d 0.0 0.0 0.0\n" %(id))
        
        f.write("\n")
        f.close()

if __name__ == "__main__":
    main()
