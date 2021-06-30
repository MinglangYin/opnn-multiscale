"""
Author: Minglang Yin, minglang_yin@brown.edu
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import scipy.io as io

global eps 
eps = sys.float_info.epsilon

def extract_data(dataset, x_lim, y_lim):
    ## extract data between lims
    data_inner = []
    for data_tmp in dataset:
        t, x, y, u = data_tmp[0], data_tmp[1], data_tmp[2], data_tmp[3]
        if (x > x_lim[0]-eps and x < x_lim[1]+eps) and (y > y_lim[0]-eps and y < y_lim[1]+eps):
            data_inner.append(data_tmp)

    data_inner = np.array(data_inner)
    
    ## get boundary value
    u_bot = []
    u_top = []
    u_left = []
    u_right = []

    for data_tmp in data_inner:
        x, y = data_tmp[1], data_tmp[2]
        if abs(x - x_lim[0]) <= eps:
            u_left.append(data_tmp)
        if abs(x - x_lim[1]) <= eps:
            u_right.append(data_tmp)
        if abs(y - y_lim[0]) <= eps:
            u_bot.append(data_tmp)
        if abs(y - y_lim[1]) <= eps:
            u_top.append(data_tmp)

    u_bot = np.array(u_bot)
    u_bot = u_bot[np.argsort(u_bot[:, 1])]

    u_top = np.array(u_top)
    u_top = u_top[np.argsort(u_top[:, 1])]

    u_left = np.array(u_left)
    u_left = u_left[np.argsort(u_left[:, 2])]

    u_right = np.array(u_right)
    u_right = u_right[np.argsort(u_right[:, 2])]

    u_bound = np.concatenate((u_top, u_bot, u_left, u_right), axis=0)[:, 3][None, :]
    u_bound = np.repeat(u_bound, data_inner.shape[0], axis = 0)

    print(u_top)
    print(u_bot)
    print(u_left)
    print(u_right)
    
    return data_inner, u_bound

def main():
    x_lim = [0.25, 0.75]
    y_lim = [0.25, 0.75]
    id_start = 0
    id_end = 10

    for id in range(id_start, id_end+1):
        path = './data/txyu_' + str(id) + '.dat'
        data = np.loadtxt(path)

        ## extract useful data
        data, u_bound = extract_data(data, x_lim, y_lim)

        if id == 0:
            branch_input = u_bound
            trunk_input = data[:, 1:3]
            output = data[:, 3:4]
        else:
            branch_input = np.concatenate((branch_input, u_bound), axis=0)
            trunk_input = np.concatenate((trunk_input, data[:, 1:3]), axis=0)
            output = np.concatenate((output, data[:, 3:4]), axis=0)

        dim = data.shape[0]

    dict = {}
    dict['branch_input'] = branch_input
    dict['trunk_input'] = trunk_input
    dict['output'] = output
    dict['dim'] = dim

    io.savemat('dataset.mat', dict)

if __name__ == "__main__":
    main()