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
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf

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

def data_loader(datafile):
    dataset = io.loadmat(datafile)
    r = 0.3
    cir_og = [0, 0.5]
    num_b = 30 # number of points on the boundary

    ## data loading
    X = dataset["coord_init"] # reference coordinate
    x = dataset["coord"] # current coordinate
    snapshot = dataset["snapshot"] ## c_S[1], c_S[2], c_S[4], c_nn, c_E[1], c_E[2], c_E[4]
    y = snapshot[:, :, 1] # Cauchy stress 22
    dx = x - X # displacement
    num_samp = snapshot.shape[0]

    ## data in omega 2
    mask = ( (X[:, 0] - cir_og[0])**2 + (X[:, 1] - cir_og[1])**2) <= r**2
    dx_in = dx[:, mask] # displacement field
    y_in = y[:, mask]  # stress field
    X_in = X[mask]     # reference coordinate
    num_in = X_in.shape[0]

    ## boundary coordinate
    del_r = np.linspace(np.pi/2, -np.pi/2, num_b)[:, None]
    x_b = np.concatenate((np.cos(del_r)*r, np.sin(del_r)*r), axis=1)
    np.savetxt("x_b.dat", x_b)

    ## interpolation for boundary displacement
    db_x, db_y = [], []
    intp = 10
    for i in range(0, num_samp):
        # print(dx[i, :, 0].shape, X[:, 1].shape)
        fx = Rbf(X[::intp, 0], X[::intp, 1], dx[i, ::intp, 0])
        fy = Rbf(X[::intp, 0], X[::intp, 1], dx[i, ::intp, 1])
        db_x_tmp = fx(x_b[:, 0], x_b[:, 1]) # boundary displacement in x
        db_y_tmp = fy(x_b[:, 0], x_b[:, 1]) # boundary displacement in x
        db_x.append(db_x_tmp)
        db_y.append(db_y_tmp)
    db_x = np.array(db_x)
    db_y = np.array(db_y)
    db = np.concatenate((db_x, db_y), axis=1)

    ## 
    x_bran = np.zeros((num_samp*num_in, num_b*2))
    x_trun = np.zeros((num_samp*num_in, 2))
    u_g = np.zeros((num_samp*num_in, 1))
    for i in range(num_samp):
        x_bran[i*num_in:(i+1)*num_in, :] = db[i]
        x_trun[i*num_in:(i+1)*num_in, :] = X_in[:, 0:2]
        u_g[i*num_in:(i+1)*num_in, :] = y_in[i][:, None]

    return x_bran, x_trun, u_g, num_in, X_in

if __name__ == "__main__":
    _, _, _, _ = data_loader("./hyper_notch_100_nocrac.mat")


