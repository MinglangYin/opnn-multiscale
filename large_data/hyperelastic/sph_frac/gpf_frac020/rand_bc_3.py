import numpy as np
import matplotlib.pyplot as plt
import os 
import random

# num_pt = 100
num_sample = 100
# x = np.linspace(0, 1, num_pt)

with open('rand_disp_exp.txt') as f:
    lines = f.readlines()

os.makedirs("new_input", exist_ok=True)

for id in range(0, num_sample):
    ## read and write
    new_file = "new_input/case_" + str(id) 
    os.makedirs(new_file, exist_ok=True)
    f_write = new_file + "/in.hgo_frac"
    with open(f_write, 'w') as fnew:
        with open("in.hgo_frac") as f:
            while True:
                currentline = f.readline()
                words = currentline.split()
                if 'vel_up' in words: 
                    words[2] = "atom"
                    words[3] = lines[id] # write the expression
                    fnew.write(" ".join(words))
                    fnew.write('\n')
                else:
                    fnew.write(currentline)
                if not currentline: 
                    break
            fnew.write('\n')
    print(f"{id+1}/{num_sample} file is done.")
