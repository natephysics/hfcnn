# %%
from hfcnn import files
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib

# %%
data_path = '../data/raw/df.pkl'

df = files.import_file_from_local_cache(data_path)

# %%
sns.set(style='whitegrid', rc={"grid.linewidth": 0.1})
sns.set_context("paper", font_scale=2)                                                  
color = sns.color_palette("Set2", 6)
dims = [30, 30]
fig, ax = plt.subplots(figsize=dims)

sns.boxplot(
    ax=ax,
    x="program_num", 
    y="I_tor_A", 
    data=df, 
    palette=color, 
    whis=np.inf, 
    width=0.5, 
    linewidth = 0.7
    )
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
sns.despine(left=True, bottom=True)
plt.savefig('boxplot.png')   

# %%
integrated_HL = []
file_path = "../data/raw/51/"
for row in df.iterrows(figsize=(15, 15)):
    image = files.import_file_from_local_cache(file_path + str(row[1]['times']) + '.pkl')
    integrated_HL.append(sum(image.flatten()))

# %%
matplotlib.rcParams.update({'font.size': 22})
file_path = "../data/raw/51/"
IAs = df['I_A'].unique()
for i in range(len(IAs)):
    temp = df[df['I_A'] == IAs[i]]
    index = temp['int_heat_load'].argmax()
    temp = temp.reset_index(drop=True)
    name = temp['times'][index]
    current_image = files.import_file_from_local_cache(file_path + str(name) + '.pkl')
    fig, ax = plt.subplots(figsize=[30,7])
    im = ax.imshow(current_image.clip(min=0))
    plt.title('Program Num: ' + str(temp['program_num'][index]) + ' large heat load example')
    fig.colorbar(im)
    plt.savefig(str(temp['program_num'][index])+'_large.png')
    plt.show()
    

# %%
matplotlib.rcParams.update({'font.size': 22})
file_path = "../data/raw/51/"
IAs = df['I_A'].unique()
for i in range(10,14):
    temp = df[df['I_A'] == IAs[19]]
    index = i
    temp = temp.reset_index(drop=True)
    name = temp['times'][index]
    current_image = files.import_file_from_local_cache(file_path + str(name) + '.pkl')
    fig, ax = plt.subplots(figsize=[30,7])
    im = ax.imshow(current_image.clip(min=0))
    plt.title('Program Num: ' + str(temp['program_num'][index]) + ' small heat load example')
    fig.colorbar(im)
    # plt.savefig(str(temp['program_num'][index])+'_small.png')
    plt.show()
    
# %%
fig, ax = plt.subplots(figsize=[30,7])
im = ax.imshow(current_image.clip(min=0))
plt.title('Program Num: ' + str(temp['program_num'][index]) + ' small heat load example')
fig.colorbar(im)
plt.savefig(str(temp['program_num'][index])+'_small.png')
plt.show()
# %%
sns.set(style='whitegrid', rc={"grid.linewidth": 1})
sns.set_context("paper", font_scale=5)                                                  
color = sns.color_palette("Set2", 6)
dims = [30, 30]
fig, ax = plt.subplots(figsize=dims)

sns.scatterplot(
    ax=ax,
    x="program_num", 
    y="I_A", 
    data=df, 
    palette=color,  
    linewidth = 0.7
    )
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
sns.despine(left=True, bottom=True)
plt.savefig('scatterplot.png')   
# %%
file_path = "../data/raw/51/"
temp = df[df['program_num'] == '20180927.33']
temp = temp.reset_index(drop=True)
for i in [16, 500, 1500]:
    name = temp['times'][i]
    current_image = files.import_file_from_local_cache(file_path + str(name) + '.pkl')
    fig, ax = plt.subplots(figsize=[30,7])
    im = ax.imshow(current_image.clip(min=0))
    plt.title(' I_tor: ' + str(temp['I_tor_A'][i]))
    fig.colorbar(im)
    plt.savefig(str(temp['I_tor_A'][i])+'_for_20180927.33.png')
    plt.show()
    
# %%
