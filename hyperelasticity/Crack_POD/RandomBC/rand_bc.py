import numpy as np
import matplotlib.pyplot as plt
import os 
import random

random.seed(10)

def fourier(x, a, b, c, d):
    y = c
    for id, _ in enumerate(a):
        y += a[id]*(np.sin(x*3*(id+1) + d[id]) + 1) + b[id]*(np.cos(x*3*(id+1) + d[id]) + 1)
    return y

num_pt = 100
num_sample = 100
x = np.linspace(0, 1, num_pt)
data = np.zeros((num_pt, num_sample+1))
data[:, 0] = x
f_scale = 0.001
f_string = []

# os.makedirs("new_input", exist_ok=True)

for id in range(0, num_sample):
    num_mode = np.random.randint(1, 4)
    a = np.random.rand(num_mode)*f_scale
    b = np.random.rand(num_mode)*f_scale
    c = np.random.random()*f_scale
    d = np.random.rand(num_mode)

    sensor_values = fourier(x, a, b, c, d)
    data[:, id+1] = sensor_values

    ## figure
    #fig = plt.figure(constrained_layout=False, figsize=(5, 5))
    #gs = fig.add_gridspec(1, 1)
    #ax = fig.add_subplot(gs[0])
    #h = ax.plot(sensor_values, linewidth=3)
    #name = "disp_" + str(id) + ".png"
    #fig.savefig(name)
    #plt.close()

    ## string 
    tmp_series = "{:5f}".format(c)
    for ii, _ in enumerate(a):
        # tmp_series += "a1*(sin(x*n+d1)+1) + b1*(cos(x*n+d1)+1)"
        # print(tmp_series)
        # exit()
        tmp_series += "+{:6f}*sin(x*{:d}+{:3f})+{:6f}*cos(x*{:d}+{:3f})".format(a[ii], 3*(ii+1), d[ii], b[ii], 3*(ii+1), d[ii]) 
    #tmp_series = "exp(-time)*(" + tmp_series + ")"
    f_string.append(tmp_series)

    ## read and write
    new_file = "new_input/case_" + str(id) 
    os.makedirs(new_file, exist_ok=True)
    f_write = new_file + "/in.hgo"
    with open(f_write, 'w') as fnew:
        with open("in.hgo") as f:
            while True:
                currentline = f.readline()
                words = currentline.split()
                if 'vel_up' in words: 
                    words[2] = "atom"
                    words[3] = tmp_series
                    fnew.write(" ".join(words))
                    fnew.write('\n')
                else:
                    fnew.write(currentline)
                if not currentline: 
                    break
            fnew.write('\n')
    print(f"{id+1}/{num_sample} file is done.")
