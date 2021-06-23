
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Main dependencies
import numpy
import scipy.fftpack

def fftind(size):
    k_ind = numpy.mgrid[:size, :size] - int( (size + 1)/2 )
    k_ind = scipy.fftpack.fftshift(k_ind)
    return(k_ind)

def gaussian_random_field(alpha = 3.0,
                          size = 128, 
                          flag_normalize = True):
        
    # Defines momentum indices
    k_idx = fftind(size)

    # Defines the amplitude as a power law 1/|k|^(alpha/2)
    amplitude = numpy.power( k_idx[0]**2 + k_idx[1]**2 + 1e-10, -alpha/4.0 )
    amplitude[0,0] = 0
    
    # Draws a complex gaussian random noise with normal (circular) distribution
    noise = numpy.random.normal(size = (size, size)) + 1j * numpy.random.normal(size = (size, size))
    
    # To real space
    gfield = numpy.fft.ifft2(noise * amplitude).real
    
    # Sets the standard deviation to one
    if flag_normalize:
        gfield = gfield - numpy.mean(gfield)
        gfield = gfield/numpy.std(gfield)
        
    return gfield

def main():
    import matplotlib
    import matplotlib.pyplot as plt
    example = gaussian_random_field(alpha = 5.0, size=64)
    plt.imshow(example, cmap='gray')
    plt.show()
    
if __name__ == '__main__':
    main()