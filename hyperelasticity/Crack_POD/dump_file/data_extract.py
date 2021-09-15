## notice that the order of atom id may be different from timesteps.

import numpy as np
import os
import scipy.io as io
import glob
import sys

os.makedirs("dataExtracted", exist_ok=True)

# step_id = 0
# step = []
snapshot = []
coord = []

# id_list = []
id_start = 0
id_end = 99
first_step = 0
last_step = 15000
for ii in range(id_start, id_end+1):
    fname = "dump_"+str(ii)+".LAMMPS"
    print(f"reading {ii}/{id_end}")
    with open(fname,'r') as f:
        currentline = f.readline()
        while True:
            if not currentline: break
            words = currentline.split()
            if 'TIMESTEP' in words: 
                currentline = f.readline()
                cur_step = int(currentline.split()[0])
                if cur_step == last_step:
                    ## get number of atoms
                    currentline = f.readline()
                    currentline = f.readline()
                    num_atoms = int(currentline.split()[0])
                    for _ in range(0, 5):
                        currentline = f.readline()

                    tmp = []
                    id_tmp = []
                    coord_tmp = []
                    for id in range(0, num_atoms):
                        currentline = f.readline()
                        words = currentline.split()
                        words = np.array(words).astype(np.float)
                        
                        coord_tmp.append(words[2:5]) # coord
                        id_tmp.append(words[0])
                        tmp.append(words[8:15]) # c_S[1], c_S[2], c_S[4], c_nn, c_E[1], c_E[2], c_E[4]

                    sort_id = np.array(id_tmp).argsort()
                    coord_tmp = np.array(coord_tmp)
                    tmp = np.array(tmp)
                    snapshot.append(tmp[sort_id])
                    coord.append(coord_tmp[sort_id])
                    
            else:
                currentline = f.readline()

## read initial coordinate           
with open("dump_0.LAMMPS",'r') as f:
    currentline = f.readline()
    while True:
        if not currentline: break
        words = currentline.split()
        if 'TIMESTEP' in words: 
            currentline = f.readline()
            cur_step = int(currentline.split()[0])
            if cur_step == first_step:
                ## get number of atoms
                currentline = f.readline()
                currentline = f.readline()
                num_atoms = int(currentline.split()[0])
                for _ in range(0, 5):
                    currentline = f.readline()

                tmp = []
                id_tmp = []
                coord_init = []
                for id in range(0, num_atoms):
                    currentline = f.readline()
                    words = currentline.split()
                    words = np.array(words).astype(np.float)

                    coord_init.append(words[2:5]) # coord
                    id_tmp.append(words[0])
                    
                sort_id = np.array(id_tmp).argsort()
                coord_init = np.array(coord_init)[sort_id]
                break  
        else:
            currentline = f.readline()    
            
dict = {}
snapshot = np.array(snapshot)
coord = np.array(coord)
# id_list = np.array(id_list)
# print( (id_list[0] - id_list[-1]) )
# np.savetxt("check.dat", id_list[0] - id_list[-1] )

dict["coord_init"] = coord_init
dict["snapshot"] = snapshot
dict["coord"] = coord
print(coord.shape)
print(snapshot.shape)
io.savemat("dataset.mat", dict)