import numpy as np
import matplotlib.pyplot as plt
from spaces import *

def main():
    space = GRF(2, length_scale=0.2, N=17, interp="cubic")

    features = space.random(1)
    x = np.linspace(0, 1, num=17)[:, None]
    y = np.linspace(0, 1, num=17)[:, None]
    xx, yy = np.meshgrid(x, y)
    coord = np.concatenate((xx.flatten()[:, None], yy.flatten()[:, None]), axis=1)
    print(coord)
    sensor_values = space.eval_u(features, coord)

    print(sensor_values)
    exit()

    fig = plt.figure(constrained_layout=False, figsize=(4, 4))
    gs = fig.add_gridspec(1, 1)

    ax = fig.add_subplot(gs[0,0])
    print(space)
    exit()
    h = ax.imshow(space, origin='lower')
    # ax.axes.xaxis.set_visible(False)
    # ax.axes.yaxis.set_visible(False)
    # h.set_clim(vmin=0, vmax=1)
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    # fig.colorbar(h, ax=ax, cax=cax)

    fig.savefig('/random_f.png')
    plt.close()


if __name__ == "__main__":
    main()