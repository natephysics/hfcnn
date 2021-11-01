# %%
# loading required packages
import os
from re import A
from typing import Callable, Union, List
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from hfcnn import files
import random

# Defined a class to make it a bit easier to use the hydra instantiate. 

class Filter():
    def __init__(
        self,
        fn_req_imgs: bool=False, 
        raw_img_folder: str=False, 
        **kwargs
        ):
        """A filter object used to generate filter functions.

        Args:
            raw_img_folder (str): path to raw image folder
            filter (str): filter function to be applied
            fn_req_imgs (bool): indicates if the filter needs an image path
        """
        self.fn_req_imgs = fn_req_imgs
        self.raw_img_folder = raw_img_folder
        self.kwargs = kwargs
        

    def set_img_path(self, new_raw_img_folder: str):
        """If the raw_img_folder is not False, will add raw_img_folder to kwargs/

        Args:
            raw_img_folder (str): path to the raw data folder
        """
        if self.fn_req_imgs:
            self.raw_img_folder = new_raw_img_folder

    def row_filter(self) -> Callable:
        """Returns a function that takes a row from a data frame as an input
        and returns True or False.

        If a raw_img_folder is not None, will return a function 

        Returns:
            [Callable]
        """                
        return None


################################
# helper functions for filters #
################################

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


def load_and_filter(row: pd.Series, raw_img_folder):
    """Function designed to make it easier to apply filters that require images
    to a pandas dataframe.

    Args:
        row (pd.Series): the row expected from df.apply()

        raw_img_folder (str): path to the directory containing the images
    """
    path = files.generate_file_path(row['times'], row['port'], raw_img_folder)
    image = files.import_file_from_local_cache(path)
    return image

def check_file_available(row: pd.Series, raw_img_folder):
    path = files.generate_file_path(row['times'], row['port'], raw_img_folder)
    return os.path.exists(path)


####################################
# filters at the data sample level #
####################################

# %%
class Data_Selection_Filter(Filter):
    def __init__(
        self, 
        raw_img_folder: str = False, 
        int_threshold: float = 5, 
        **kwargs
        ):
        super().__init__(
            fn_req_imgs=True, 
            raw_img_folder=raw_img_folder, 
            **kwargs
            )
        self.raw_img_folder = raw_img_folder
        self.int_threshold = int_threshold


    def filter_fn(self, row: pd.Series):
        """A filter developed to exclude image data with too little heat load

        Args:
            image (np.ndarray): head load image
            raw_img_folder (str): path to the directory containing the images
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

        # load the image file from the pandas row
        image = load_and_filter(row, self.raw_img_folder)
        
        # remove values below zero convert to int
        current_image = image.astype(int).clip(min=0)

        # integrate
        current_image_con = conv2d(current_image, kernel, kernel_size)
        integrated = np.sum(
            current_image_con[current_image_con > ignore_below_this_value]/1e8
            )

        # check to see if integral is above a threshold to consider the data
        if integrated > self.int_threshold:
            return True
        else:
            return False

    def row_filter(self) -> Callable:
        """Returns a function that takes a row from a data frame as an input
        and returns True or False.

        If a raw_img_folder is not None, will return a function 

        Returns:
            [Callable]
        """                
        return lambda x: self.filter_fn(x)


class Missing_Data_Filter(Filter):
    def __init__(
        self, 
        raw_img_folder: str = False,  
        **kwargs
        ):
        super().__init__(
            fn_req_imgs=True, 
            raw_img_folder=raw_img_folder, 
            **kwargs
            )
        self.raw_img_folder = raw_img_folder


    def filter_fn(self, row: pd.Series):
        """Filters if the file is unavailable."""
        return check_file_available(row, self.raw_img_folder)


    def row_filter(self) -> Callable:
        """Returns a function that takes a row from a data frame as an input
        and returns True or False.

        If a raw_img_folder is not None, will return a function 

        Returns:
            [Callable]
        """                
        return lambda x: self.filter_fn(x)


#######################################
# Filters at the program number level #
#######################################

def split(prog_num_list: list, ratio_list: Union[float, list]):
    """Takes in a list of program numbers and creates sublists based on the 
    proportions list. Numbers expected to be between (0-1). If one number, x, is
    provided, will return two sets of ratios x, and 1-x. If two are provided, 
    will return three sets of ratios x, y, and 1 - x - y. 

    Args:
        prog_num_list (list): List of the program numbers to split
            
        ratio_list (list): proportion of the data to split for each sub list. 
            Numbers expected to be between (0-1).
    """
    if "PYTHONHASHSEED" in os.environ:
        random.seed(int(os.environ["PYTHONHASHSEED" ]))

    if isinstance(ratio_list, float):
        ratio_list = [ratio_list]

    # return error if any element of ratio list are outside of the interval 0-1
    for i in ratio_list:
        if (i <= 0) or (i >= 1):
            raise ValueError("Ratio list elements must be (0-1)")

    # get the number of elements
    list_length = len(prog_num_list)
    
    # shuffle the list
    random.shuffle(prog_num_list)

    # determine the index that bounds each sublist.
    sub_index = [0]
    for ratio in ratio_list:
        sub_index.append(round(list_length * ratio) + sum(sub_index))
    
    sub_index.append(list_length)
    
    # return the list of list of sublists
    return [prog_num_list[sub_index[i] : sub_index[i+1]] for i in range(len(sub_index) - 1)]