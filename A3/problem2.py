import numpy as np
from scipy.ndimage import convolve, maximum_filter
import cv2


def gauss2d(sigma, fsize):
    """ Create a 2D Gaussian filter

    Args:
        sigma: width of the Gaussian filter
        fsize: (w, h) dimensions of the filter
    Returns:
        *normalized* Gaussian filter as (h, w) np.array
    """
    m, n = fsize
    x = np.arange(-m / 2 + 0.5, m / 2)
    y = np.arange(-n / 2 + 0.5, n / 2)
    xx, yy = np.meshgrid(x, y, sparse=True)
    g = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return g / np.sum(g)


def derivative_filters():
    """ Create derivative filters for x and y direction

    Returns:
        fx: derivative filter in x direction
        fy: derivative filter in y direction
    """
    fx = np.array([[0.5, 0, -0.5]])
    fy = fx.transpose()
    return fx, fy


def compute_hessian(img, gauss, fx, fy):
    """ Compute elements of the Hessian matrix

    Args:
        img:
        gauss: Gaussian filter
        fx: derivative filter in x direction
        fy: derivative filter in y direction

    Returns:
        I_xx: (h, w) np.array of 2nd derivatives in x direction
        I_yy: (h, w) np.array of 2nd derivatives in y direction
        I_xy: (h, w) np.array of 2nd derivatives in x-y direction
    """
    #
    # You code here
    #



def compute_criterion(I_xx, I_yy, I_xy, sigma):
    """ Compute criterion function

    Args:
        I_xx: (h, w) np.array of 2nd derivatives in x direction
        I_yy: (h, w) np.array of 2nd derivatives in y direction
        I_xy: (h, w) np.array of 2nd derivatives in x-y direction
        sigma: scaling factor

    Returns:
        criterion: (h, w) np.array of scaled determinant of Hessian matrix
    """
    #
    # You code here
    #


def nonmaxsuppression(criterion, threshold):
    """ Apply non-maximum suppression to criterion values
        and return Hessian interest points

        Args:
            criterion: (h, w) np.array of criterion function values
            threshold: criterion threshold
        Returns:
            rows: (n,) np.array with y-positions of interest points
            cols: (n,) np.array with x-positions of interest points
    """

    #
    # You code here
    #


def imagepatch_descriptors(gray, rows, cols):
    """ Get image patch descriptors for every interes point

        Args:
            img: (h, w) np.array with image gray values
            rows: (n,) np.array with y-positions of interest points
            cols: (n,) np.array with x-positions of interest points
        Returns:
            descriptors: (n, patch_size**2) np.array with image patch feature descriptors
    """

    #
    # You code here
    #


def match_interest_points(descriptors1, descriptors2):
    """ Brute-force match the interest points descriptors of two images using the cv2.BFMatcher function. 
    Select a reasonable distance measurement to be used and set "crossCheck=True".

    Args:
        descriptors1: (n, patch_size**2) np.array with image patch feature descriptors
        descriptors2: (n, patch_size**2) np.array with image patch feature descriptors
    Returns:
        matches: (m) list of matched descriptor pairs
    """
    #
    # You code here
    #
