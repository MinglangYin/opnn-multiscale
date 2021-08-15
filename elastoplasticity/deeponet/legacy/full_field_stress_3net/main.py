"""
Author: Minglang Yin, minglang_yin@brown.edu
Branch Net Input:       [u(x1), u(x2), ..., u(xn)]
Trunk Net Input:        [x, y]
Output:                 [s_xx, s_yy, s_xy, PEEQ]
"""
import os
import torch
import time
import numpy as np
import scipy.io as io

from utils import *
from opnn import *
from scipy.interpolate import Rbf
# from scipy.interpolate import interp2d

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def main():
    ## hyperparameters
    args = ParseArgument()
    epochs = args.epochs
    device = args.device
    save_step = args.save_step
    restart = args.restart

    dim_br = [80, 100, 100, 100]
    dim_tr = [2, 100, 100, 100]
    lmd = 1e-6

    num_train = 900
    num_test = 100
    intv = 3 ## sample interval at boundary
    datafile = "../plastic_dataset_full.mat"

    ## create folders
    dump_train = './Predictions/Train/'
    dump_test = './Predictions/Test/'
    model_path = 'CheckPts/model_chkpts.pt'
    os.makedirs(dump_train, exist_ok=True)
    os.makedirs(dump_test, exist_ok=True)
    os.makedirs('CheckPts', exist_ok=True)
    
    ## dataset
    x_b, x_t, y, dim = extract_data(datafile, intv)
    out_dim = y.shape[1]

    y_train = y[:dim*num_train]
    x_b_train = x_b[:dim*num_train]
    x_t_train = x_t[:dim*num_train]

    y_test = y[dim*num_train:dim*(num_train+num_test)]
    x_b_test = x_b[dim*num_train:dim*(num_train+num_test)]
    x_t_test = x_t[dim*num_train:dim*(num_train+num_test)]

    ## tensor
    y_train_tensor = torch.tensor(y_train, dtype=torch.float).to(device)
    x_b_train_tensor = torch.tensor(x_b_train, dtype=torch.float).to(device)
    x_t_train_tensor = torch.tensor(x_t_train, dtype=torch.float).to(device)

    y_test_tensor = torch.tensor(y_test, dtype=torch.float).to(device)
    x_b_test_tensor = torch.tensor(x_b_test, dtype=torch.float).to(device)
    x_t_test_tensor = torch.tensor(x_t_test, dtype=torch.float).to(device)


    ## initialization
    model = opnn(dim_br, dim_tr, out_dim).to(device)
    model = model.float()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    train_loss = np.zeros((args.epochs, 1))
    test_loss = np.zeros((args.epochs, 1))

    ## training
    def train(epoch, x_b_batch, x_t_batch, y_batch):
        model.train()
        def closure():
            optimizer.zero_grad()
            y_out = model(x_b_batch, x_t_batch)

            ## L2 regularization
            regularization_loss = 0
            # for param in model._branch.parameters():
            #     regularization_loss += torch.sum(param**2)
            # for param in model._trunk.parameters():
            #     regularization_loss += torch.sum(param**2)
            # for param in model._out_layer.parameters():
            #     regularization_loss += torch.sum(param**2)

            loss_data = ((y_out - y_batch)**2).mean() + lmd*regularization_loss

            loss = loss_data
            train_loss[epoch, 0] = loss
            loss.backward()
            return loss
        optimizer.step(closure)

    ## restart
    if restart == 1:
        print('Restart!', flush=True)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    ## Iterations
    print('start training...', flush=True)
    tic = time.time()
    for epoch in range(0, epochs):
        train(epoch, x_b_train_tensor, x_t_train_tensor, y_train_tensor)

        ## testing error
        y_pred = to_numpy(model.forward(x_b_test_tensor, x_t_test_tensor))
        y_true = to_numpy(y_test_tensor)
        test_loss[epoch, 0] = ((y_pred - y_true)**2).mean()
        print(f'Epoch: {epoch+1}, Train Loss: {train_loss[epoch, 0]:.6f}, Test Loss: {test_loss[epoch, 0]:.6f}', flush=True)

        # print(f'Epoch: {epoch+1}, Train Loss: {train_loss[epoch, 0]:.6f}', flush=True)

        ## Save model
        if (epoch+1)%save_step == 0: 
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, model_path)

    toc = time.time()

    print(f'total training time: {int((toc-tic)/60)} min', flush=True)
    np.savetxt('./train_loss.txt', train_loss)
    np.savetxt('./test_loss.txt', test_loss)

    ## plot loss function
    num_epoch = train_loss.shape[0]
    x = np.linspace(1, num_epoch, num_epoch)
    
    ## plotting
    fig = plt.figure(constrained_layout=False, figsize=(6, 6))
    gs = fig.add_gridspec(1, 1)
    ax = fig.add_subplot(gs[0])
    ax.plot(x, train_loss, color='blue', label='Training Loss')
    ax.plot(x, test_loss, color='red', label='Test Loss', linestyle='dashed')

    ax.set_yscale('log')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epochs')
    ax.legend(loc='upper left')
    fig.savefig('./loss_his.png')

    ## save train
    y_train_pred = model.forward(x_b_train_tensor, x_t_train_tensor)
    y_pred = to_numpy(y_train_pred)
    y_true = y_train

    ## plotting mesh
    # dim_plot = 101
    # x = np.linspace(-0.35, 0.35, dim_plot)
    # y = np.linspace(-0.35, 0.35, dim_plot)
    
    # xx, yy = np.meshgrid(x, y)
    # xx = xx.flatten()
    # yy = yy.flatten()
    # r_out = 0.3
    # r_in = 0.1

    # out_bdry = (xx**2+yy**2) > (r_out**2-(1E-5))
    # in_bdry = (xx**2+yy**2) < (r_in**2-(1E-5))
    # id_out_bdry = [i for i, val in enumerate(out_bdry) if val]
    # id_in_bdry = [i for i, val in enumerate(in_bdry) if val]

    for i in range(0, 10):
        x1_plot = x_t_train[dim*i:dim*(i+1), 0]
        x2_plot = x_t_train[dim*i:dim*(i+1), 1]
        y_pred_plot = y_pred[dim*i:dim*(i+1)]
        y_true_plot = y_true[dim*i:dim*(i+1)]

        ### scatter
        print(f"printing case: {i}")
        fig = plt.figure(constrained_layout=False, figsize=(14, 14))
        gs = fig.add_gridspec(3, 3)

        for j in range(0, 3):
            ## sigma_xx
            ax = fig.add_subplot(gs[j*3])
            h = ax.scatter(x1_plot, x2_plot, s=20, c=y_true_plot[:, j])
            # h.set_clim(vmin=-4, vmax=4)
            ax.set_title("true")
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(h, ax=ax, cax=cax)

            ax = fig.add_subplot(gs[j*3+1])
            h = ax.scatter(x1_plot, x2_plot, s=20, c=y_pred_plot[:, j])
            # h.set_clim(vmin=-4, vmax=4)
            ax.set_title("prediction")
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(h, ax=ax, cax=cax)

            ax = fig.add_subplot(gs[j*3+2])
            h = ax.scatter(x1_plot, x2_plot, s=20, c=abs(y_true_plot[:, j]-y_pred_plot[:, j]))
            # h.set_clim(vmin=0, vmax=0.4)
            ax.set_title("abs err")
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(h, ax=ax, cax=cax)

        fig.savefig(dump_train + '/img_scatter_' + str(i) + '.png')
        plt.close()
    
    ## save train
    y_test_pred = model.forward(x_b_test_tensor, x_t_test_tensor)
    y_pred = to_numpy(y_test_pred)
    y_true = y_test

    for i in range(0, 10):
        x1_plot = x_t_test[dim*i:dim*(i+1), 0]
        x2_plot = x_t_test[dim*i:dim*(i+1), 1]
        y_pred_plot = y_pred[dim*i:dim*(i+1)]
        y_true_plot = y_true[dim*i:dim*(i+1)]

        ### scatter
        print(f"printing Testing case: {i}")
        fig = plt.figure(constrained_layout=False, figsize=(14, 14))
        gs = fig.add_gridspec(3, 3)

        for j in range(0, 3):
            ## sigma_xx
            ax = fig.add_subplot(gs[j*3])
            h = ax.scatter(x1_plot, x2_plot, s=20, c=y_true_plot[:, j])
            # h.set_clim(vmin=-4, vmax=4)
            ax.set_title("true")
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(h, ax=ax, cax=cax)

            ax = fig.add_subplot(gs[j*3+1])
            h = ax.scatter(x1_plot, x2_plot, s=20, c=y_pred_plot[:, j])
            # h.set_clim(vmin=-4, vmax=4)
            ax.set_title("prediction")
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(h, ax=ax, cax=cax)

            ax = fig.add_subplot(gs[j*3+2])
            h = ax.scatter(x1_plot, x2_plot, s=20, c=abs(y_true_plot[:, j]-y_pred_plot[:, j]))
            # h.set_clim(vmin=0, vmax=0.4)
            ax.set_title("abs err")
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(h, ax=ax, cax=cax)

        fig.savefig(dump_test + '/img_scatter_' + str(i) + '.png')
        plt.close()

if __name__ == "__main__":
    main()
