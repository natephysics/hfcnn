# Needs a lot of revisions 

# %%
# loading required packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns
import torch
import torch.nn.functional as F
import doctest
from lib import files


# %%
# import the data
df = files.import_file_from_local_cache('.\data\df.hkl')


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
    # convert the inputs into pytorch tensor objects
    image = torch.tensor(np.expand_dims(image, axis=(0,1)))
    kernel = torch.tensor(np.expand_dims(kernel, axis=(0,1)))

    # Convolve the arrays
    return F.conv2d(image, kernel, stride=strides, padding=0).numpy().squeeze()

doctest.testmod()


# %%

# Get unique list of program numbers and only take every 6th number
# program_nums = df['program_num'].unique()[::6]
program_nums = df['program_num'].unique()


# how many programs to pull from
num_of_programs = program_nums.size

# number of data points to take per program
num_of_data = 29

##########################################################
##### Important parameters that impact the selection ##### 
# bins for histograms
bin_num = 20

# only count integral values above this threshold 
int_threshold = 5
##########################################################

# array to contain the index of the first datapoint to exceed the threshold.
# included_index = np.ones(num_of_programs) * -1
included_index = -1

# array to contain histogram data
current_heatmaps = np.empty((num_of_data, bin_num))

# loop over all the programs
# for program_index, program_num in enumerate(program_nums):
program_num = program_nums[3]

# select a program number
df_temp = df[df['program_num'] == program_num]

# index to use to identify which data point exceeded the integral threshold. 
included_index = -1

# loop over a subset of the data
for row in df_temp[1:num_of_data + 1].reset_index().iterrows():
    current_index = row[0]

    current_image = files.import_file_from_local_cache(
        files.generate_file_path(
            row[1]['times'], 
            row[1]['port']
            )
        )
    
    # remove values below zero convert to int
    current_image = current_image.astype(int).clip(min=0)

    # construct the hisogram
    y_values, bin_edges = np.histogram(
        conv2d(current_image, kernel, 24).flatten(),
        bins=20,
        range=(range_min, range_max)
        )

    # integrate
    current_image_con = conv2d(current_image, kernel, 24)
    integrated = np.sum(
        current_image_con[current_image_con > ignore_below_this_value]/1e8
        )

    # check to see if integral is above a threshold to consider the data
    if integrated > int_threshold and included_index == -1:
        included_index = current_index

    current_heatmaps[current_index] = y_values
        # get his patches
# for program_index, program_num in enumerate(program_nums):
#         current_heatmaps[program_index, current_index] = y_values

# construct the heatmap
column_labels = [str(j/2) for j in range(1, 21)]
heat_mapped = pd.DataFrame(current_heatmaps + 1, columns=column_labels)
fig = plt.figure(figsize=(15, 15))
res = sns.heatmap(
    heat_mapped + 1, 
    linewidths = 2, 
    linecolor = "white", 
    norm=LogNorm(),
    cbar_kws={'label': 'Histogram Bar Height'}
    )
plt.xlabel('Histogram x interval', fontsize=20)
plt.ylabel('Heat Load Image Index', fontsize=20)
plt.title(f'Program Number: {program_num}', fontsize=20)
# include a green line to indicate which values meet the threshold
if included_index >= 0:
    res.axhline(y = included_index, color = 'g', linewidth = 3)
plt.show()


# %%
axes = plt.figure(constrained_layout=True).subplot_mosaic(
                [['bar', 'bar', 'bar'], # Note repitition of 'bar'
                 ['hist', '.', 'scatter']])
for k, ax in axes.items():
    ax.text(0.5, 0.5, k, ha='center', va='center', 
            fontsize=36, color='magenta')
# Using dictionary to change subplot properties
axes['bar'].set_title('A bar plot', fontsize=24)    
axes['hist'].set_title('A histogram', fontsize=24)    
axes['scatter'].set_title('A scatter plot', fontsize=24)


# %%
def construct_plot_mosaic(labels: list, max_row_len: int = 3):
    """Takes a list of strings and formats them into array of strings with a 
    row length no longer than max_row_len. Any empty values will be replaced 
    with the string '.'.

    Args:
        labels (list(str)): axes labels
        max_row_len (int): [description]
    
    >>> construct_plot_mosaic(['1', '2', '3', '4'], 2)
    np.array([['1', '2'], ['3', '4']], dtype='<U1')

    >>> construct_plot_mosaic(['1', '2', '3', '4'], 3)
    np.array([['1', '2', '3'], ['4', '.', '.']], dtype='<U1')
    """
    length = np.size(labels)
    if not length % max_row_len == 0:
        labels.extend(['.'] * (max_row_len - (length % max_row_len)))
        length = np.size(labels)
    return np.reshape(labels, (int(length/max_row_len), max_row_len))



# %%
np.array_equal(
    construct_plot_mosaic(['1', '2', '3', '4'], 2), 
    np.array([['1', '2'], ['3', '4']], dtype='<U1')
)


# %%
