"""
Author: Minglang Yin, minglang_yin@brown.edu
"""
import numpy as np
import scipy
import torch
import argparse
import scipy.io as io
import matplotlib.pyplot as plt
# from scipy.interpolate import UnivariateSpline
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression, Ridge, Lasso

def ParseArgument():
    parser = argparse.ArgumentParser(description='DeepONet')
    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                        help = 'number of epochs to train (default: 1000)')
    parser.add_argument('--device', type=str, default='cuda', metavar='N',
                        help = 'computing device (default: GPU)')
    parser.add_argument('--save-step', type=int, default=10000, metavar='N',
                        help = 'save_step (default: 10000)')
    parser.add_argument('--restart', type=int, default=0, metavar='N',
                        help = 'if restart (default: 0)')
    parser.add_argument('--batch-size', type=int, default=1000000, metavar='N',
                        help = 'batch_size (default: 1000000)')

    args = parser.parse_args()
    return args

def to_numpy(input):
    if isinstance(input, torch.Tensor):
        return input.detach().cpu().numpy()
    elif isinstance(input, np.ndarray):
        return input
    else:
        raise TypeError('Unknown type of input, expected torch.Tensor or '\
            'np.ndarray, but got {}'.format(type(input)))

def extract_data(datafile, intv, intcoord):
    dataset = io.loadmat(datafile)
    
    ## output (stress)
    y = dataset["sigma"][:, ::intcoord, 0:1]
    # y = np.concatenate((y1, y2), axis=2)
    dim_case = y.shape[1]
    y[219, :, :] = y[220, :, :]
    y = y.reshape(-1, 1)

    ## trunk net input (coordinate)
    x_t = dataset["xy"][:, ::intcoord, :]
    x_t[219, :, :] = x_t[220, :, :]
    x_t = x_t.reshape(-1, 2)

    ## branch net input (displacement)
    x_b_tmp = dataset["disp_bdry"][:, ::intv, :]
    x_b_tmp[219, :, :] = x_b_tmp[220, :, :]
    x_b_tmp = x_b_tmp.reshape((x_b_tmp.shape[0], -1), order='F')
    
    x_b = np.zeros((x_t.shape[0], x_b_tmp.shape[1]))

    for i in range(0, x_b_tmp.shape[0]):
        x_b[i*dim_case:(i+1)*dim_case, :] = np.tile(x_b_tmp[i, :], (dim_case, 1))

    ## coordinate at the interface
    x_intf = dataset["xy_bdry"][::intv]
    np.savetxt("x_intf_ext.dat", x_intf)

    ## scaling factor (disp: 100, stress: 10)
    y = y*1e1
    x_b = x_b*1e2

    return x_b, x_t, y, dim_case

if __name__ == "__main__":
    _, _, _ = extract_data("../plastic_dataset_full_more.mat", 3)


