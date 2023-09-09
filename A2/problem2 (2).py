from copy import deepcopy
import numpy as np
from PIL import Image
from scipy.special import binom
from scipy.ndimage import convolve

import matplotlib.pyplot as plt

def loadimg(path):
    """ Load image file

    Args:
        path: path to image file
    Returns:
        image as (H, W) np.array normalized to [0, 1]
    """
    #
    # You code here     
    #
    img = plt.imread(path)
    return img

def gauss2d(sigma, fsize):
    """ Create a 2D Gaussian filter

    Args:
        sigma: width of the Gaussian filter
        fsize: (W, H) dimensions of the filter
    Returns:
        *normalized* Gaussian filter as (H, W) np.array
    """
    #
    # You code here     
    #
    w, h = fsize
    g_filter = np.zeros((h,w), dtype=float)
    for y in range(h):
        for x in range(w):
            # change center to (0,0)
            xx = x - (w-1)/2
            yy = y - (h-1)/2
            g_filter[y, x] = (1 / (2*np.pi*np.square(sigma))) * np.exp(- (np.square(xx) + np.square(yy)) / (2 * np.square(sigma)))
    # normalize
    sum = np.sum(g_filter)
    g_filter_normalized = g_filter/sum
    return g_filter_normalized

def binomial2d(fsize):
    """ Create a 2D binomial filter

    Args:
        fsize: (W, H) dimensions of the filter
    Returns:
        *normalized* binomial filter as (H, W) np.array
    """
    #
    # You code here
    #
    w, h = fsize # assumption on w==h
    filter1 = np.zeros((1,w), dtype=float)
    
    N = w-1
    for k in range(N+1): # 0 to N
        filter1[0, k] = binom(N, k)
    # normalize
    filter1_sum = np.sum(filter1)
    filter1 = filter1/filter1_sum
    
    filter2 = filter1.copy().reshape((h,1))
    return filter2 @ filter1

def downsample2(img, f):
    """ Downsample image by a factor of 2
    Filter with Gaussian filter then take every other row/column

    Args:
        img: image to downsample
        f: 2d filter kernel
    Returns:
        downsampled image as (H, W) np.array
    """
    #
    # You code here
    #
    h, w = img.shape
    h_ds = (h-1)//2 + 1
    w_ds = (w-1)//2 + 1
    img_ds = np.zeros((h_ds, w_ds), dtype = float)
    # 1) filter
    img_filtered = convolve(img, f, mode = 'nearest')
    # 2) discard
    for y in range(h_ds):
        for x in range(w_ds):
            img_ds[y, x] = img_filtered[2*y, 2*x]
    return img_ds     

def upsample2(img, f):
    """ Upsample image by factor of 2

    Args:
        img: image to upsample
        f: 2d filter kernel
    Returns:
        upsampled image as (H, W) np.array
    """
    #
    # You code here
    #
    h, w = img.shape
    h_us = h * 2
    w_us = w * 2
    img_us = np.zeros((h_us, w_us), dtype = float)
    # 1) add rows and cols
    for y in range(h):
        for x in range(w):
            img_us[2*y, 2*x] = img[y, x]
    # 2) filter & scale 4
    return 4 * convolve(img_us, f, mode = 'nearest')

def gaussianpyramid(img, nlevel, f):
    """ Build Gaussian pyramid from image

    Args:
        img: input image for Gaussian pyramid
        nlevel: number of pyramid levels
        f: 2d filter kernel
    Returns:
        Gaussian pyramid, pyramid levels as (H, W) np.array
        in a list sorted from fine to coarse
    """
    #
    # You code here
    #
    gpyramid = [] # list of image
    gpyramid.append(img) # level 0
    img_nextLevel = img
    for i in range(nlevel-1): # level 1 to nlevel-1
        img_nextLevel = downsample2(img_nextLevel, f)
        gpyramid.append(img_nextLevel)
    return gpyramid

def laplacianpyramid(gpyramid, f):
    """ Build Laplacian pyramid from Gaussian pyramid

    Args:
        gpyramid: Gaussian pyramid
        f: 2d filter kernel
    Returns:
        Laplacian pyramid, pyramid levels as (H, W) np.array
        in a list sorted from fine to coarse
    """
    #
    # You code here
    #
    lpyramid = []
    nlevel = len(gpyramid)
    for i in range(nlevel-1): # 0 to nlevel-2
        img_l = gpyramid[i] - upsample2(gpyramid[i+1], f)
        lpyramid.append(img_l)
    # nlevel-1
    lpyramid.append(gpyramid[nlevel-1])
    return lpyramid

def reconstructimage(lpyramid, f):
    """ Reconstruct image from Laplacian pyramid

    Args:
        lpyramid: Laplacian pyramid
        f: 2d filter kernel
    Returns:
        Reconstructed image as (H, W) np.array clipped to [0, 1]
    """
    #
    # You code here
    #     


def amplifyhighfreq(lpyramid, l0_factor=1.3, l1_factor=1.6):
    """ Amplify frequencies of the finest two layers of the Laplacian pyramid

    Args:
        lpyramid: Laplacian pyramid
        l0_factor: amplification factor for the finest pyramid level
        l1_factor: amplification factor for the second finest pyramid level
    Returns:
        Amplified Laplacian pyramid, data format like input pyramid
    """
    #
    # You code here
    #   


def createcompositeimage(pyramid):
    """ Create composite image from image pyramid
    Arrange from finest to coarsest image left to right, pad images with
    zeros on the bottom to match the hight of the finest pyramid level.
    Normalize each pyramid level individually before adding to the composite image

    Args:
        pyramid: image pyramid
    Returns:
        composite image as (H, W) np.array
    """
    #
    # You code here
    #   
    nlevel = len(pyramid)
    # for image size (background)
    h, _ = pyramid[0].shape
    w = 0
    for i in range(nlevel):
        _, w_i = pyramid[i].shape
        w += w_i
    img_cpst = np.zeros((h, w), dtype=float)
    # normalize [0,1] and add each images
    w_stt = 0 # start place of new image in width
    for i in range(nlevel):
        img_i = pyramid[i]
        h_i, w_i = img_i.shape
        # normalize
        img_i_normalized = (img_i - np.min(img_i))/(np.max(img_i) - np.min(img_i))
        for y in range(h_i):
            for x in range(w_i):
                img_cpst[y, x + w_stt] = img_i_normalized[y, x]
        w_stt += w_i
    
    return img_cpst