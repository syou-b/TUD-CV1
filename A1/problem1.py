import numpy as np
import matplotlib.pyplot as plt
import numpy as np


def display_image(img):
    """ Show an image with matplotlib:

    Args:
        Image as numpy array (H,W,3)
    """
    plt.imshow(img)
    plt.show()


def save_as_npy(path, img):
    """ Save the image array as a .npy file:

    Args:
        Image as numpy array (H,W,3)
    """

    np.save(path, img)


def load_npy(path):
    """ Load and return the .npy file:

    Args:
        Path of the .npy file
    Returns:
        Image as numpy array (H,W,3)
    """

    img = np.load(path)
    return(img)

def mirror_horizontal(img):
    """ Create and return a horizontally mirrored image:

    Args:
        Loaded image as numpy array (H,W,3)

    Returns:
        A horizontally mirrored numpy array (H,W,3).
    """

    flipped_img = np.fliplr(img)
    return(flipped_img)


def display_images(img1, img2):
    """ display the normal and the mirrored image in one plot:

    Args:
        Two image numpy arrays
    """
    figure = plt.figure()

    figure.add_subplot(1, 2, 1)
    plt.imshow(img1)
    plt.title("original")

    figure.add_subplot(1, 2, 2)
    plt.imshow(img2)
    plt.title("flipped")

    plt.show()