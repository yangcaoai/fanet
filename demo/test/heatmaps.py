import numpy as np
import seaborn as sns
from  pyecharts.charts import Bar
from  pyecharts.charts import Bar3D
import os
import glob
from pyecharts import options as opts
import pandas as pd
import random
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import  cm

def colormap():
  cdict = ['#FFFFFF', '#9ff113', '#5fbb44', '#f5f329', '#e50b32']
  return colors.ListedColormap(cdict, 'indexed')

def show(score, nums, base_name):
    stages = ['1', '2', '3', '4']
    mods = ['bod', 'seg']
    dilation_rates = ["pooling", "1x1", "3x3-6", "3x3-12", "3x3-18", "3x3-24", "3x3-36"]
    xy = [dilation_rates, stages]
    
    data = {}
    mod = 'bod'
#     mod = 'seg'
    for i in range(4):
        tmp = {}
        for j in range(7):
            tmp[dilation_rates[j]] = score['stage' + stages[i]][mod][0][j]
        data[stages[i]] = tmp
    
    sns.set(font_scale=2.3)
    pd_data = pd.DataFrame(data)
    my_cmap = colormap()

    plt.figure(figsize=(10, 9))
    ax = sns.heatmap(pd_data, cmap='YlGnBu', annot=True, fmt=".3f") #"rainbow" tab20c YlGnBu
    #ax.set_title('Boundary')#
    ax.set_title('Segmentation')
    plt.yticks(rotation=55) 
    plt.xlabel('Stage')
    #plt.ylabel('Kernel Setting')
    plt.rc('font',family='Times New Roman')
    fig = ax.get_figure()
    fig.savefig("{}_{}.pdf".format(base_name, mod))
    #fig.savefig("try.pdf")

def find(file_list, nums, base_name): 

    #print(file_list)
    for idx, i in enumerate(file_list):
        score = np.load(i, allow_pickle=True).item()
        if(idx==0):
            res = score
        else:
            for stage in stages:
                for mod in mods:
                    res[stage][mod] = res[stage][mod] + score[stage][mod]
                    
    for stage in stages:
        for mod in mods:
            res[stage][mod] = ((res[stage][mod] / nums))#.round(3))
    
    show(res, nums, base_name)
        #print(res)
        #if(idx==2):
        #    break
        


if __name__ == '__main__':
    stages = ['stage1', 'stage2', 'stage3', 'stage4']
    mods = ['seg', 'bod']

    file_list = sorted(glob.glob('scores/*.npy'))
    all_list = [False] #, False]

    for all_or_not in all_list:
        if(all_or_not):
            nums = len(file_list)
            find(file_list, nums, None)
        else:
            idx_get_list = [0] #1, 500, 1000, 1500, 2000, 32, 740, 1780]
            for idx_get in idx_get_list:
                file_list_temp = file_list[idx_get:(idx_get+1)]
                base_name = os.path.basename(file_list_temp[0])[:-4]
                find(file_list_temp, 1, base_name)