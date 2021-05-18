# %%
# loading required packages
from re import A
import numpy as np
import pandas as pd
import torch
from torch._C import BoolType
import torch.nn.functional as F
from hfcnn.lib import files


# %%
# helper functions
def conv2d(image, kernel, strides):
    """Uses PyTorch to convolve a kernel over a 2D image
    
    >>> image = np.array([[1,2,3], [4,5,6], [7,8,9], [10,11,12]])
    >>> kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1],])
    >>> conv2d(image, kernel, 2)
    array(45)
    
    >>> conv2d(image, kernel, 1)
    array([45, 72])
    """
    # convert the inputs into PyTorch tensor objects
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
def load_and_filter(row: pd.Series, img_dir='./data/raw'):
    """Function designed to make it easier to apply data_selection() to a pandas
    dataframe.

    Args:
        row (pd.Series): the row expected from df.apply()

        img_dir (str): path to the directory containing the images
        
    Returns:
       Boolean : if the data that corresponds to the row is included by the filter.
    """
    path = files.generate_file_path(row['times'], row['port'], img_dir)
    image = files.import_file_from_local_cache(path)
    return data_selection(image)


# %%
def return_filter(filter_names: str, *args):
    """Takes in the name of a filter and the filters arguments and returns that
    filter.

    Args:
        filter_name ([type]): [description]
    """
    if filter_names == "data_selection":
        return lambda x: load_and_filter(x, *args)
