"""
python extract_pointdata.py dump.LAMMPS <id>
"""
import numpy as np
import os
import glob
import sys

# Usage
if ( len(sys.argv) < 3 ):
    raise Exception('Usage: python file.py file.data n_concentration')

id = int(sys.argv[2])		# number of species

# IMPORTANT : the full file path must be enclosed in single quote
fnames = glob.glob(str(sys.argv[1]))
newfnames = ['SMD_id' + str(id) + ".dat"]

def emptyline(thisline):        # check whether thisline is empty.
    if not (len(thisline.split()) == 0):
        raise Exception("Not an empty line.")

for ct, fn in enumerate(fnames):
    with open(newfnames[ct],'w') as fnew:
        with open(fn) as f:
            while True:
                currentline = f.readline()
                if not currentline: break
                words = currentline.split()
                if words[0] == str(id):
                    fnew.write(" ".join(words))
                    fnew.write('\n')
