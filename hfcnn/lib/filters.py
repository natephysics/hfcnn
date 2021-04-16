# %%
# loading required packages
import numpy as np
import torch
import torch.nn.functional as F


# %%
# helper functions
def conv2d(image, kernel, strides):
    """Uses pytorch to convolve a kernel over a 2D image
    
    >>> image = np.array([[1,2,3], [4,5,6], [7,8,9], [10,11,12]])
    >>> kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1],])
    >>> conv2d(image, kernel, 2)
    array(45)
    
    >>> conv2d(image, kernel, 1)
    array([45, 72])
    """
    # convert the inputs into pytorch tensor objects
    image = torch.tensor(np.expand_dims(image, axis=(0,1)))
    kernel = torch.tensor(np.expand_dims(kernel, axis=(0,1)))

    # Convolve the arrays
    return F.conv2d(image, kernel, stride=strides, padding=0).numpy().squeeze()


# %%
def data_selection(image: np.ndarray, int_threshold=5):
    """[summary]

    Args:
        image (np.ndarray): [description]
        int_threshold (int, optional): [description]. Defaults to 5.

    Returns:
        [type]: [description]
    """

    ##########################################################
    ##### Important parameters that impact the selection ##### 
    # def the size of the kernel
    kernel_size = 24

    # construct a kernel to convolve over the images
    kernel = np.ones((kernel_size, kernel_size), dtype=int) # kernel size

    # when integrating ignore data below this bin
    ignore_below_this_value = 7e7
    ##########################################################
    # remove values below zero convert to int
    current_image = image.astype(int).clip(min=0)


    # integrate
    current_image_con = conv2d(current_image, kernel, kernel_size)
    integrated = np.sum(
        current_image_con[current_image_con > ignore_below_this_value]/1e8
        )

    # check to see if integral is above a threshold to consider the data
    if integrated > int_threshold:
        return True
    else:
        return False

# %%
import hickle as hkl
