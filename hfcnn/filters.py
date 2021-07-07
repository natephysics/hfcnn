# %%
# loading required packages
from re import A
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from hfcnn import files
import random


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
    """A filter developed to exclude image data with too little heat load

    Args:
        image (np.ndarray): head load image
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
def zero_neg_filter(image: np.ndarray):
    """Zeros out all negative values

    Args:
        image (np.ndarray): Expects image data.

    Returns:
        (np.ndarray): zeroed image
    """
    return image.clip(min=0)


# %%
def sum_image(image: np.ndarray):
    """Sums the values of all the pixels
    

    Args:
        image (np.ndarray): [description]

    Returns:
        [type]: [description]
    """


# %%
def load_and_filter(row: pd.Series, filter, img_dir='./data/raw'):
    """Function designed to make it easier to apply filters that require images
    to a pandas dataframe.

    Args:
        row (pd.Series): the row expected from df.apply()

        img_dir (str): path to the directory containing the images
    """
    path = files.generate_file_path(row['times'], row['port'], img_dir)
    image = files.import_file_from_local_cache(path)
    return filter(image)


# %%
def return_filter(filter_name: str, *args):
    """Takes in the name of a filter and the filters arguments and returns that
    filter.

    Args:
        filter_name ([type]): [description]
    """
    if filter_name == "data_selection":
        return lambda x: load_and_filter(x, data_selection, *args)


# %%
def split(prog_num_list: list, ratio_list: list, seed: int=4):
    """Takes in a list of program numbers and creates sublists based on the 
    proportions list. Numbers expected to be between (0-1). If one number, x, is
    provided, will return two sets of ratios x, and 1-x. If two are provided, 
    will return three sets of ratios x, y, and 1 - x - y. 

    Args:
        prog_num_list (list): List of the program numbers to split
            
        ratio_list (list): proportion of the data to split for each sub list. 
            Numbers expected to be between (0-1).

        seed (int): random seed.
    """
    # return error if any element of ratio list are outside of the interval 0-1
    for i in ratio_list:
        if (i <= 0) or (i >= 1):
            raise ValueError("Ratio list elements must be (0-1)")

    # get the number of elements
    list_length = len(prog_num_list)
    
    # shuffle the list
    random.seed(seed)
    random.shuffle(prog_num_list)

    # determine the index that bounds each sublist.
    sub_index = [0]
    for ratio in ratio_list:
        sub_index.append(round(list_length * ratio) + sum(sub_index))
    
    sub_index.append(list_length)
    
    # return the list of list of sublists
    return [prog_num_list[sub_index[i] : sub_index[i+1]] for i in range(len(sub_index) - 1)]