import numpy as np
from scipy.ndimage import convolve


def loaddata(path):
    """ Load bayerdata from file

    Args:
        Path of the .npy file
    Returns:
        Bayer data as numpy array (H,W)
    """

    img = np.load(path)
    return(img)
    


def separatechannels(bayerdata):
    """ Separate bayer data into RGB channels so that
    each color channel retains only the respective
    values given by the bayer pattern and missing values
    are filled with zero

    Args:
        Numpy array containing bayer data (H,W)
    Returns:
        red, green, and blue channel as numpy array (H,W)
    """
    r = np.zeros(np.shape(bayerdata))
    g = np.zeros(np.shape(bayerdata))
    b = np.zeros(np.shape(bayerdata))

    for row in range(len(bayerdata)):
        if row % 2 == 0:
            r[row][1::2] = bayerdata[row][1::2]
            g[row][0::2] = bayerdata[row][0::2]
        else:
            g[row][1::2] = bayerdata[row][1::2]
            b[row][0::2] = bayerdata[row][0::2]
    
    return(r, g, b)

            

    


def assembleimage(r, g, b):
    """ Assemble separate channels into image

    Args:
        red, green, blue color channels as numpy array (H,W)
    Returns:
        Image as numpy array (H,W,3)
    """
    
    return(r + g + b)


def interpolate(r, g, b):
    """ Interpolate missing values in the bayer pattern
    by using bilinear interpolation

    Args:
        red, green, blue color channels as numpy array (H,W)
    Returns:
        Interpolated image as numpy array (H,W,3)
    """
    convolve_mode="mirror"
    kernel_rb = np.array([[0.25,0.5,0.25],[0.5,1.0,0.5],[0.25,0.5,0.25]])
    kernel_g = np.array([[0,0.25,0],[0.25,1.0,0.25],[0,0.25,0]])
    return(convolve(r, kernel_rb, mode=convolve_mode)+convolve(g, kernel_g, mode=convolve_mode)+convolve(b, kernel_rb, mode=convolve_mode))
    
