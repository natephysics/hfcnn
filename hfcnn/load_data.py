# %%
# loading required packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as pltd
from matplotlib.colors import LogNorm
import tensorflow as tf
import seaborn as sns
import torch
import torch.nn.functional as F
from hfcnn.lib import files


# %%
# import the data
df = files.import_file_from_local_cache('..\data\df.hkl')

# %%
# helper functions

# # Create a dummy generator.
# def generate_features(df: pd.DataFrame, directory: str='../data'):
#     # Function to generate a single data point from the dataframe
#     for row in df.iterrows():
#         timestamp, port = row[1]['times'], row[1]['port']
#         image_path = files.generate_file_path(timestamp, port, directory)
#         image = files.import_file_from_local_cache(image_path)
#         yield image, row[1]['PC1']


# # %%
# # Load data using tf data api using the generator
# data = tf.data.Dataset.from_generator(lambda: generate_features(df), output_types=(tf.float32, tf.float32))


# # Batch data (aggregate records together).
# data = data.batch(batch_size=4)
# # Prefetch batch (pre-load batch for faster consumption).
# data = data.prefetch(buffer_size=1)


# # %%
# # Display data.
# for batch_str, batch_vector in data.take(5):
#     print(batch_str, batch_vector)
# # %%
