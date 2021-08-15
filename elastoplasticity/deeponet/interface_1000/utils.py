"""
Author: Minglang Yin, minglang_yin@brown.edu
"""
import numpy as np
import scipy
import torch
import argparse
import scipy.io as io
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf
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

def extract_data(datafile, intv, add_extra_data=1):
    dataset = io.loadmat(datafile)

    ## displacement, stress, and coordinate
    y = dataset["txy_bdry"]
    x_t = dataset["xy_bdry"]
    x_b_tmp = dataset["disp_bdry"]

    dim_case = int(y.shape[1]/intv)

    ## add extra data
    if add_extra_data == 1:
        r0 = 0.3 ### hard coding
        dataset_e = np.load("../AbaqusOutput_Job-Example_PlaneStrainPlasticity.npy")
        x_e = dataset_e[:, 1:3]
        u_e = dataset_e[:, 3:5]
        sigma_e = dataset_e[:, 10:14]

        fx_fem_e = Rbf(x_e[:, 0], x_e[:, 1], u_e[:, 0])
        fy_fem_e = Rbf(x_e[:, 0], x_e[:, 1], u_e[:, 1])
        ux_fem_e = fx_fem_e(x_t[:, 0], x_t[:, 1])[:, None]
        uy_fem_e = fy_fem_e(x_t[:, 0], x_t[:, 1])[:, None]
        u_fem_e = np.concatenate((ux_fem_e, uy_fem_e), axis=1 )

        fxx_e = Rbf(x_e[:, 0], x_e[:, 1], sigma_e[:, 0])
        fyy_e = Rbf(x_e[:, 0], x_e[:, 1], sigma_e[:, 1])
        fxy_e = Rbf(x_e[:, 0], x_e[:, 1], sigma_e[:, 3])
        sxx_e_intf = fxx_e(x_t[:, 0], x_t[:, 1])
        syy_e_intf = fyy_e(x_t[:, 0], x_t[:, 1])
        sxy_e_intf = fxy_e(x_t[:, 0], x_t[:, 1])

        nx = x_t[:, 0]/r0
        ny = x_t[:, 1]/r0
        tx_e = sxx_e_intf*nx+sxy_e_intf*ny
        ty_e = sxy_e_intf*nx+syy_e_intf*ny
        txy_e = np.concatenate( ( tx_e[:, None], ty_e[:, None] ), axis=1 )

        y[219, :, :] = txy_e
        x_b_tmp[219, :, :] = u_fem_e

    else:
        y[219, :, :] = y[220, :, :]
        x_b_tmp[219, :, :] = x_b_tmp[220, :, :]

    # reshape and tile
    y = y[:, ::intv].reshape(-1, 2)
    x_t_tmp = x_t.reshape(-1, 2)[::intv]
    x_b_tmp = x_b_tmp[:, ::intv].reshape((x_b_tmp.shape[0], -1), order='F')

    x_b = np.zeros((y.shape[0], x_b_tmp.shape[1]))
    x_t = np.zeros((y.shape[0], x_t_tmp.shape[1]))

    for i in range(0, x_b_tmp.shape[0]): # for each case with every interface points
        x_b[i*dim_case:(i+1)*dim_case, :] = np.tile(x_b_tmp[i, :], (dim_case, 1))
        x_t[i*dim_case:(i+1)*dim_case, :] = x_t_tmp

    ## coordinate at the interface
    np.savetxt("x_intf_ext.dat", x_t_tmp)

    ## scaling factor (100)
    y = y*1e2
    x_b = x_b*1e2

    return x_b, x_t, y

if __name__ == "__main__":
    # _, _, _ = extract_data("plastic_dataset.mat", 1)
    x_b, x_t, y = extract_data("../plastic_dataset.mat", 3)


