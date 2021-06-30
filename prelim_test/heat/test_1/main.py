"""
Author: Minglang Yin, minglang_yin@brown.edu
DeepONet for solving Poisson equation
"""

import os
import torch
import time
import numpy as np
import scipy.io as io

from utils import *
from opnn import *
from scipy.interpolate import Rbf
from scipy import interpolate

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def main():
    ## hyperparameters
    args = ParseArgument()
    epochs = args.epochs
    device = args.device
    save_step = args.save_step
    restart = args.restart

    dim_br = [68, 100, 100, 100]
    dim_tr = [2, 100, 100, 100]
    lmd = 0 #1e-5

    num_cases = 11
    num_train = 11
    num_test = 0
    
    ## create folders
    dump_test = './Predictions/Test/'
    dump_train = './Predictions/Train/'
    model_path = 'CheckPts/model_chkpts.pt'
    os.makedirs(dump_train, exist_ok=True)
    os.makedirs(dump_test, exist_ok=True)
    os.makedirs('CheckPts', exist_ok=True)
    
    ## dataset
    dataset = io.loadmat('dataset.mat')
    y = dataset["output"]
    x_b = dataset["branch_input"]
    x_t = dataset["trunk_input"]
    dim = int(dataset["dim"])

    y_train = y[:dim*num_train]
    x_b_train = x_b[:dim*num_train]
    x_t_train = x_t[:dim*num_train]

    print(x_t.shape)
    print(x_t_train.shape)
    # y_test = y[dim*num_train:]
    # x_b_test = x_b[dim*num_train:]
    # x_t_test = x_t[dim*num_train:]

    # print(y_train.shape)
    # print(x_b_train.shape)
    # print(x_t_train.shape)
    # print(y_test.shape)
    # print(x_b_test.shape)
    # print(x_t_test.shape)

    ##
    y_train_tensor = torch.tensor(y_train, dtype=torch.float).to(device)
    x_b_train_tensor = torch.tensor(x_b_train, dtype=torch.float).to(device)
    x_t_train_tensor = torch.tensor(x_t_train, dtype=torch.float).to(device)

    # y_test_tensor = torch.tensor(y_test, dtype=torch.float).to(device)
    # x_b_test_tensor = torch.tensor(x_b_test, dtype=torch.float).to(device)
    # x_t_test_tensor = torch.tensor(x_t_test, dtype=torch.float).to(device)

    ## initialization
    model = opnn(dim_br, dim_tr).to(device)
    model = model.float()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    train_loss = np.zeros((args.epochs, 1))
    test_loss = np.zeros((args.epochs, 1))

    ## training
    def train(epoch, x_b, x_t, y):
        model.train()
        def closure():
            optimizer.zero_grad()
            y_out = model(x_b, x_t)

            ## L2 regularization
            #regularization_loss = 0
            # for param in model._flatten.parameters():
            #     regularization_loss += torch.sum(param**2)
            #for param in model._trunk.parameters():
                #regularization_loss += torch.sum(param**2)
            #for param in model._out_layer.parameters():
                #regularization_loss += torch.sum(param**2)

            loss_data = ((y_out - y)**2).mean()

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
        ## Training
        # if epoch == 10000:
        #     optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)
        # elif epoch == epochs - 1000:
        #     optimizer = torch.optim.LBFGS(model.parameters())

        train(epoch, x_b_train_tensor, x_t_train_tensor, y_train_tensor)
        print(f'Epoch: {epoch+1}, Train Loss: {train_loss[epoch, 0]:.6f}', flush=True)

        ## Testing
        # y_pred = to_numpy(model.forward(x_b_test_tensor, x_t_test_tensor))
        # y_true = to_numpy(y_test_tensor)
        # test_loss[epoch, 0] = ((y_pred - y_true)**2).mean()
        # print(f'Epoch: {epoch+1}, Train Loss: {train_loss[epoch, 0]:.6f}, Test Loss: {test_loss[epoch, 0]:.6f}', flush=True)

        ## Save model
        if (epoch+1)%save_step == 0: 
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, model_path)

    toc = time.time()

    print(f'total training time: {int((toc-tic)/60)} min', flush=True)
    np.savetxt('./train_loss.txt', train_loss)
    # np.savetxt('./test_loss.txt', test_loss)

#################################################################
    # dim_x = int(np.sqrt(dim))
    # y_test_pred = model.forward(x_b_test_tensor, x_t_test_tensor)
    # y_pred = to_numpy(y_test_pred)[:, 0]
    # y_true = y_test[:, 0]

    # x1_new = np.linspace(0.25, 0.75, 17)
    # x2_new = np.linspace(0.25, 0.75, 17)
    # xx1, xx2 = np.meshgrid(x1_new, x2_new)
    # xx1 = xx1.reshape(-1)
    # xx2 = xx2.reshape(-1)

    # count = 0
    # for i in range(0, num_test):
    #     x1_plot = x_t_test[dim*i:dim*(i+1), 0]
    #     x2_plot = x_t_test[dim*i:dim*(i+1), 1]
    #     y_pred_plot = y_pred[dim*i:dim*(i+1)]
    #     y_true_plot = y_true[dim*i:dim*(i+1)]

    #     # print(x1_plot)
    #     # print(x2_plot)
    #     # exit()

    #     # f = Rbf(x1_plot, x2_plot, y_pred_plot, epsilon=2)
    #     # y_pred_intp = f(xx1, xx2).reshape(dim_x, dim_x)

    #     # # y_pred_intp = rbf(xx1, xx2)

    #     # f = Rbf(x1_plot, x2_plot, y_true_plot, epsilon=2)
    #     # y_true_intp = f(xx1, xx2).reshape(dim_x, dim_x)
        
    #     print(f"printing case: {i}")
    #     fig = plt.figure(constrained_layout=False, figsize=(14, 4))
    #     gs = fig.add_gridspec(1, 3)

    #     ax = fig.add_subplot(gs[0])
    #     # h = ax.imshow(y_pred_intp)
    #     h = ax.scatter(x1_plot, x2_plot, c=y_pred_plot,  marker='o', s=100)

    #     ax.set_title('Testing case (Pred): ' + str(i))
    #     # h.set_clim(vmin=0, vmax=1)
    #     divider = make_axes_locatable(ax)
    #     cax = divider.append_axes("right", size="5%", pad=0.05)
    #     fig.colorbar(h, ax=ax, cax=cax)

    #     ax = fig.add_subplot(gs[1])
    #     # h = ax.imshow(y_true_intp)
    #     h = ax.scatter(x1_plot, x2_plot, c=y_true_plot,  marker='o', s=100)

    #     ax.set_title('Testing case (True): ' + str(i))
    #     # h.set_clim(vmin=0, vmax=1)
    #     divider = make_axes_locatable(ax)
    #     cax = divider.append_axes("right", size="5%", pad=0.05)
    #     fig.colorbar(h, ax=ax, cax=cax)
        
    #     ax = fig.add_subplot(gs[2])
    #     h = ax.scatter(x1_plot, x2_plot, c=abs(y_pred_plot - y_true_plot),  marker='o', s=100)
    #     ax.set_title('Testing case (Err): ' + str(i))
    #     # h.set_clim(vmin=0, vmax=1)
    #     divider = make_axes_locatable(ax)
    #     cax = divider.append_axes("right", size="5%", pad=0.05)
    #     fig.colorbar(h, ax=ax, cax=cax)

    #     fig.savefig(dump_test + '/img_' + str(i) + '.png')
    #     plt.close()

    ## save train
    y_train_pred = model.forward(x_b_train_tensor, x_t_train_tensor)
    y_pred = to_numpy(y_train_pred)[:, 0]
    y_true = y_train[:, 0]

    count = 0
    for i in range(0, 10):
        x1_plot = x_t_train[dim*i:dim*(i+1), 0]
        x2_plot = x_t_train[dim*i:dim*(i+1), 1]
        y_pred_plot = y_pred[dim*i:dim*(i+1)]
        y_true_plot = y_true[dim*i:dim*(i+1)]

        # rbf = Rbf(x1_plot, x2_plot, y_pred_plot, epsilon=2)
        # y_pred_intp = rbf(xx1, xx2).reshape(dim_x, dim_x)

        # rbf = Rbf(x1_plot, x2_plot, y_true_plot, epsilon=2)
        # y_true_intp = rbf(xx1, xx2).reshape(dim_x, dim_x)
        

        print(f"printing case: {i}")
        fig = plt.figure(constrained_layout=False, figsize=(13, 4))
        gs = fig.add_gridspec(1, 3)

        ax = fig.add_subplot(gs[0])
        # h = ax.imshow(y_pred_intp)
        h = ax.scatter(x1_plot, x2_plot, c=y_pred_plot, marker='o', s=100)

        # h.set_clim(vmin=0, vmax=1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(h, ax=ax, cax=cax)

        ax = fig.add_subplot(gs[1])
        # h = ax.imshow(y_true_intp)
        h = ax.scatter(x1_plot, x2_plot, c = y_true_plot,  marker='o', s=100)
        ax.set_title('Training case (True): ' + str(i))
        # h.set_clim(vmin=0, vmax=1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(h, ax=ax, cax=cax)

        ax = fig.add_subplot(gs[2])
        # h = ax.imshow(abs(y_pred_intp - y_true_intp))
        h = ax.scatter(x1_plot, x2_plot, c=abs(y_pred_plot - y_true_plot),  marker='o', s=100)
        ax.set_title('Training case (Err): ' + str(i))
        # h.set_clim(vmin=0, vmax=1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(h, ax=ax, cax=cax)

        fig.savefig(dump_train + '/img_' + str(i) + '.png')
        plt.close()
    
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

if __name__ == "__main__":
    main()