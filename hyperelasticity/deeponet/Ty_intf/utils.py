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
    parser.add_argument('--epochs', type=int, default=50000, metavar='N',
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

def circ_noraml(x, center):
    norm = np.linalg.norm(x-center, ord=2, axis=1)[:, None]
    return (x-center)/norm

def data_loader(datafile):
    dataset = io.loadmat(datafile)
    r = 0.3
    cir_og = [0, 0.5]
    num_b = 30 # number of points on the boundary

    ## data loading
    X = dataset["coord_init"] # reference coordinate
    x = dataset["coord"] # current coordinate
    snapshot = dataset["snapshot"] ## c_P[1] (P11), c_P[2] (P12), c_P[3] (P13), c_P[4] (P21), c_P[5] (P22), c_P[6] (P23)
    y = snapshot

    dx = x - X # displacement
    num_samp = snapshot.shape[0]

    ## boundary coordinate
    d_theta = np.linspace(np.pi/2, -np.pi/2, num_b)[:, None]
    x_b = np.concatenate((np.cos(d_theta)*r+cir_og[0], np.sin(d_theta)*r+cir_og[1]), axis=1)
    N_intf = circ_noraml(x_b, cir_og)
    np.savetxt("x_intf_ext.dat", x_b)

    ## interpolation for boundary displacement and traction
    db_x, db_y = [], []
    Tx, Ty = [], []
    intp = 10
    for i in range(0, num_samp):
        ## boundary displacement 
        fx = Rbf(X[::intp, 0], X[::intp, 1], dx[i, ::intp, 0])
        fy = Rbf(X[::intp, 0], X[::intp, 1], dx[i, ::intp, 1])

        db_x_tmp = fx(x_b[:, 0], x_b[:, 1]) # boundary displacement in x
        db_y_tmp = fy(x_b[:, 0], x_b[:, 1]) # boundary displacement in x

        db_x.append(db_x_tmp)
        db_y.append(db_y_tmp)

        ## traction
        f_p11 = Rbf(X[::intp, 0], X[::intp, 1], y[i, ::intp, 0])
        f_p12 = Rbf(X[::intp, 0], X[::intp, 1], y[i, ::intp, 1])
        f_p21 = Rbf(X[::intp, 0], X[::intp, 1], y[i, ::intp, 3])
        f_p22 = Rbf(X[::intp, 0], X[::intp, 1], y[i, ::intp, 4])

        p11_intf = f_p11(x_b[:, 0], x_b[:, 1])
        p12_intf = f_p12(x_b[:, 0], x_b[:, 1])
        p21_intf = f_p21(x_b[:, 0], x_b[:, 1])
        p22_intf = f_p22(x_b[:, 0], x_b[:, 1])

        Tx_intf = p11_intf*N_intf[:, 0] + p12_intf*N_intf[:, 1]
        Ty_intf = p21_intf*N_intf[:, 0] + p22_intf*N_intf[:, 1]

        Tx.append(Tx_intf)
        Ty.append(Ty_intf)

    db_x = np.array(db_x)
    db_y = np.array(db_y)
    db = np.concatenate((db_x, db_y), axis=1)

    Tx = np.array(Tx)
    Ty = np.array(Ty)

    ## 
    x_bran = np.zeros((num_b*num_samp, num_b*2))
    x_trun = np.zeros((num_b*num_samp, 2))
    u_g = np.zeros((num_b*num_samp, 1))
    for i in range(num_samp):
        x_bran[i*num_b:(i+1)*num_b, :] = db[i]
        x_trun[i*num_b:(i+1)*num_b, :] = x_b
        u_g[i*num_b:(i+1)*num_b, :] = Ty[i][:, None] ## Tx!!

    return x_bran, x_trun, u_g, num_b, x_b

if __name__ == "__main__":
    _, _, _, _ = data_loader("./hyper_notch_100_nocrac.mat")


