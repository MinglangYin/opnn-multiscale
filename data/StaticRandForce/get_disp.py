import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from scipy.io import savemat
from numpy import save

def main():
    num_case = 5000
    dim = 17
    x1_data = []
    x2_data = []
    u_data = []
    v_data = []
    f_data = []

    for i in range(0, num_case):
        print(i)
        tmp = np.loadtxt("./new_data/uerror_"+str(i)+".csv")
        x1_data.append(tmp[:, 1])
        x2_data.append(tmp[:, 2])
        u_data.append(tmp[:, 3])

        tmp = np.loadtxt("./new_data/verror_"+str(i)+".csv")
        x = tmp[:, 1]
        y = tmp[:, 2]
        v_data.append(tmp[:, 3])

        tmp = np.loadtxt("./rand_force/rand_f_"+str(i)+".dat")
        x_f = tmp[:, 0]
        y_f = tmp[:, 1]
        f_data.append(tmp[:, 2])

        # fig = plt.figure(constrained_layout=False, figsize=(14, 4))
        # gs = fig.add_gridspec(1, 3)

        # ax = fig.add_subplot(gs[0])
        # h = ax.contourf(x.reshape(17, 17), y.reshape(17, 17), u_data[-1].reshape(17, 17))
        # h.set_clim(vmin=-0.5, vmax=0.5)
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes("right", size="5%", pad=0.05)
        # fig.colorbar(h, ax=ax, cax=cax)

        # ax = fig.add_subplot(gs[1])
        # h = ax.contourf(x.reshape(17, 17), y.reshape(17, 17), v_data[-1].reshape(17, 17))
        # h.set_clim(vmin=-0.5, vmax=0.5)
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes("right", size="5%", pad=0.05)
        # fig.colorbar(h, ax=ax, cax=cax)

        # ax = fig.add_subplot(gs[2])
        # h = ax.contourf(x_f.reshape(29, 29), y_f.reshape(29, 29), f_data[-1].reshape(29, 29))
        # h.set_clim(vmin=-2, vmax=2)
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes("right", size="5%", pad=0.05)
        # fig.colorbar(h, ax=ax, cax=cax)

        # fig.savefig("./figure/random_f_" + str(i) + ".png")
        # plt.close()

    ## generate training data triplet
    f_data = np.array(f_data).reshape(num_case, dim+12, dim+12)
    f_data = f_data[:, 6:-6, 6:-6].reshape(num_case, -1)
    f_train = np.zeros((num_case*dim*dim, dim*dim))
    for i in range(0, num_case):
        f_train[dim*dim*i:dim*dim*(i+1),:] = f_data[i:i+1, :]
    
    x1_train = np.array(x1_data).flatten()[:, None]
    x2_train = np.array(x2_data).flatten()[:, None]
    u_train = np.array(u_data).flatten()[:, None]
    v_train = np.array(v_data).flatten()[:, None]

    print(f_train.shape, x1_train.shape, x2_train.shape, u_train.shape, v_train.shape)

    dict = {"f_train": f_train, "x1_train": x1_train, 
            "x2_train": x2_train, "u_train": u_train, "v_train": v_train,
            "num_case": num_case, "dim": dim}


    save('dataset.npy', dict)

    # pickle.dump( dict, gzip.open( 'dataset.pkl.gz',   'wb' ) )

    savemat("dataset.mat", dict)

if __name__ == "__main__":
    main()