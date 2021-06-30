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
        x, y, u = data_tmp[0], data_tmp[1], data_tmp[2]
        if (x > x_lim[0]-eps and x < x_lim[1]+eps) and (y > y_lim[0]-eps and y < y_lim[1]+eps):
            data_inner.append(data_tmp)

    data_inner = np.array(data_inner)
    
    ## get boundary value
    u_bot = []
    u_top = []
    u_left = []
    u_right = []

    for data_tmp in data_inner:
        x, y = data_tmp[0], data_tmp[1]
        if abs(x - x_lim[0]) <= eps:
            u_left.append(data_tmp)
        if abs(x - x_lim[1]) <= eps:
            u_right.append(data_tmp)
        if abs(y - y_lim[0]) <= eps:
            u_bot.append(data_tmp)
        if abs(y - y_lim[1]) <= eps:
            u_top.append(data_tmp)

    u_bot = np.array(u_bot)
    u_bot = u_bot[np.argsort(u_bot[:, 0])]

    u_top = np.array(u_top)
    u_top = u_top[np.argsort(u_top[:, 0])]

    u_left = np.array(u_left)
    u_left = u_left[np.argsort(u_left[:, 1])]

    u_right = np.array(u_right)
    u_right = u_right[np.argsort(u_right[:, 1])]

    u_bound = np.concatenate((u_top, u_bot, u_left, u_right), axis=0)[:, 2][None, :]
    u_bound = np.repeat(u_bound, data_inner.shape[0], axis = 0)

    return data_inner, u_bound

def main():
    x_lim = [0.25, 0.75]
    y_lim = [0.25, 0.75]
    
    path = './data_single/xyu.dat'
    data = np.loadtxt(path)

    ## extract useful data
    data, u_bound = extract_data(data, x_lim, y_lim)

    branch_input = u_bound
    trunk_input = data[:, 0:2]
    output = data[:, 2:3]
    dim = data.shape[0]


    dict = {}
    dict['branch_input'] = branch_input
    dict['trunk_input'] = trunk_input
    dict['output'] = output
    dict['dim'] = dim

    io.savemat('dataset.mat', dict)

if __name__ == "__main__":
    main()