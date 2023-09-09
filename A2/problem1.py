import numpy as np
from scipy.ndimage import convolve


def generate_image():
    """ Generates cocentric simulated image in Figure 1.

    Returns:
        Cocentric simulated image with the size (200, 200) with increasing intesity through the center
        as np.array.
    """
    #
    # You code here.
    #
    arr = np.zeros((200,200), dtype=float)
    arr[20:180, 20:180] = 40
    arr[40:160, 40:160] = 80
    arr[60:140, 60:140] = 120
    arr[80:120, 80:120] = 160
    return arr
    

def sobel_edge(img):
    """ Applies sobel edge filter on the image to obtain gradients in x and y directions and gradient map.
    (see lecture 4 slide 57 for filter coefficients)

    Args:
        img: image to be convolved
    Returns:
        Ix derivatives of the source image in x-direction as np.array
        Iy derivatives of the source image in y-direction as np.array
        Ig gradient magnitude map computed by sqrt(Ix^2+Iy^2) for each pixel
    """
    #
    # You code here.
    #
    x_filter = 1/8 * np.array([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=float)
    y_filter = 1/8 * np.array([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=float)
    xmap = convolve(img, x_filter)
    ymap = convolve(img, y_filter)
    xymap = np.sqrt(np.square(xmap) + np.square(ymap))
    return xmap, ymap, xymap

def detect_edges(grad_map, threshold=20):
    """ Applies threshold on the edge map to detect edges.

    Args:
        grad_map: gradient map.
    Returns:
        edge_map: thresholded gradient map.
    """
    # 
    # You code here.
    #
    edge_map = grad_map.copy()
    for i in range(200):
        for j in range(200):
            edge_map[i,j] = 0 if grad_map[i, j]< threshold else grad_map[i,j]
    return edge_map

def add_noise(image, mean=0, variance=10):
    """ Applies Gaussian noise on the image.

    Args:
        img: image in np.array
        mean: mean of the noise distribution.
        variance: variance of the noise distribution.
    Returns:
        noisy_image: gaussian noise applied image.
    """
    #
    # You code here.
    #
    noisy_image = image + np.random.normal(0, np.sqrt(10), image.shape)
    return noisy_image