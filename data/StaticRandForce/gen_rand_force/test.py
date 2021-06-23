# Copyright 2017 Bruno Sciolla. All Rights Reserved.
# ==============================================================================
# Generator for 2D scale-invariant Gaussian Random Fields
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Main dependencies
import os
import numpy as np
import scipy.fftpack
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

np.random.seed(123)

def fftind(size):
    """ Returns a numpy array of shifted Fourier coordinates k_x k_y.
        
    Input args:
        size (integer): The size of the coordinate array to create
    Returns:
        k_ind, numpy array of shape (2, size, size) with:
            k_ind[0,:,:]:  k_x components
            k_ind[1,:,:]:  k_y components
            
    Example:
    
        print(fftind(5))
        
        [[[ 0  1 -3 -2 -1]
        [ 0  1 -3 -2 -1]
        [ 0  1 -3 -2 -1]
        [ 0  1 -3 -2 -1]
        [ 0  1 -3 -2 -1]]
        [[ 0  0  0  0  0]
        [ 1  1  1  1  1]
        [-3 -3 -3 -3 -3]
        [-2 -2 -2 -2 -2]
        [-1 -1 -1 -1 -1]]]
            
        """
    k_ind = np.mgrid[:size, :size] - int( (size + 1)/2 )
    k_ind = scipy.fftpack.fftshift(k_ind)
    return( k_ind )

def gaussian_random_field(alpha = 3.0,
                          size = 128, 
                          flag_normalize = True):
    """ Returns a numpy array of shifted Fourier coordinates k_x k_y.
        
    Input args:
        alpha (double, default = 3.0): 
            The power of the power-law momentum distribution
        size (integer, default = 128):
            The size of the square output Gaussian Random Fields
        flag_normalize (boolean, default = True):
            Normalizes the Gaussian Field:
                - to have an average of 0.0
                - to have a standard deviation of 1.0
    Returns:
        gfield (numpy array of shape (size, size)):
            The random gaussian random field
            
    Example:
    import matplotlib
    import matplotlib.pyplot as plt
    example = gaussian_random_field()
    plt.imshow(example)
    """
        
    # Defines momentum indices
    k_idx = fftind(size)

    # Defines the amplitude as a power law 1/|k|^(alpha/2)
    amplitude = np.power( k_idx[0]**2 + k_idx[1]**2 + 1e-10, -alpha/4.0 )
    amplitude[0,0] = 0

    # Draws a complex gaussian random noise with normal
    # (circular) distribution
    noise = np.random.normal(size = (size, size)) \
        + 1j * np.random.normal(size = (size, size))
    
    # To real space
    gfield = np.fft.ifft2(noise * amplitude).real
    
    # Sets the standard deviation to one
    if flag_normalize:
        gfield = gfield - np.mean(gfield)
        gfield = gfield/np.std(gfield)
        
    return gfield


def main():
    ##
    os.makedirs("rand_force", exist_ok=True)
    os.makedirs("figure", exist_ok=True)
    num_case = 10000

    ## 
    coord = np.loadtxt("Coords.txt")
    x = coord[:, 0]
    y = coord[:, 1]

    x_ = np.linspace(-2.74889, 2.74889, num=29)[:, None]
    y_ = np.linspace(-2.74889, 2.74889, num=29)[:, None]

    xx, yy = np.meshgrid(x_, y_)
    xx = np.transpose( xx ) ## unified with C++ code
    yy = np.transpose( yy )
    
    # coord = np.concatenate((xx.flatten()[:, None], yy.flatten()[:, None]), axis=1)
    # print(xx.flatten()-x)

    for i in range(0, num_case):
        ## 
        example = gaussian_random_field(alpha = 10.0,
                            size = 29, 
                            flag_normalize = True)/2

        data = np.concatenate((x[:, None], y[:, None], example.flatten()[:, None] ), axis=1 )
        np.savetxt("./rand_force/rand_f_" + str(i) + ".dat", data)

        fig = plt.figure(constrained_layout=False, figsize=(4.5, 4))
        gs = fig.add_gridspec(1, 1)

        ax = fig.add_subplot(gs[0, 0])
        h = ax.contourf(x.reshape(29, 29), y.reshape(29, 29), example)
        # ax.axes.xaxis.set_visible(False)
        # ax.axes.yaxis.set_visible(False)
        h.set_clim(vmin=-2, vmax=2)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(h, ax=ax, cax=cax)

        # plt.tight_layout()
        fig.savefig("./figure/random_f_" + str(i) + ".png")
        # plt.show()
        plt.close()
    
    
if __name__ == '__main__':
    main()
