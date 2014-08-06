# -*- coding: utf-8 -*-
"""
Created on Mon Aug 04 12:50:31 2014

@author: Adrián
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import matplotlib.colors as colors 


    
def compute_means(rgb_image):
    meanR = (rgb_image[:,:,0]).mean()
    meanG = (rgb_image[:,:,1]).mean()
    meanB = (rgb_image[:,:,2]).mean()  
#    print meanR, "\n", meanG, "\n", meanB, "\n"
    mean_diff = max(abs(meanR-meanG), abs(meanR-meanB), abs(meanG-meanB))
    return mean_diff
    
def compute_stds(rgb_image):
    stdR = (rgb_image[:,:,0]).std()
    stdG = (rgb_image[:,:,1]).std()
    stdB = (rgb_image[:,:,2]).std()  
    #print "STDs are:", "\n", stdR, "\n", stdG, "\n", stdB
    std_diff = max(abs(stdR-stdG), abs(stdR-stdB), abs(stdG-stdB))
    return std_diff
    
def compute_saturations(rgb_image):
    im_hsv = colors.rgb_to_hsv(rgb_image[:,:,0:3])
    # pull out just the s channel
    sat=im_hsv[:,:,1]
    return sat.mean()
    
def normalize(arr):
    arr=arr.astype('float32')
    if arr.max() > 1.0:
        arr/=255.0
    return arr    
    
pathIm1 = "Ancuti1/"
pathIm2 = "Im1/"
pathIm3 = "Ship/"
pathIm4 = "Ancuti3/"

img1=mpimg.imread(pathIm1 + "Ancuti1.png")
img2=mpimg.imread(pathIm2 + "Im1.jpg")
img3=mpimg.imread(pathIm3 + "Eustice4.jpg")
img4=mpimg.imread(pathIm4 + "Ancuti3.png")
###########################################   Me
img1_me=mpimg.imread(pathIm1 + "Ancuti1_RecoveredMe.png")
img2_me=mpimg.imread(pathIm2 + "Im1_RecoveredMe.jpg")
img3_me=mpimg.imread(pathIm3 + "Eustice4_RecoveredMe.png")
img4_me=mpimg.imread(pathIm4 + "Ancuti3_RecoveredMe.png")
###########################################   Ancuti
img1_ancuti=mpimg.imread(pathIm1 + "Ancuti1_RecoveredAncuti.png")
img2_ancuti=mpimg.imread(pathIm2 + "Im1_RecoveredAncuti.jpg")
img3_ancuti=mpimg.imread(pathIm3 + "Eustice4_RecoveredAncuti.png")
img4_ancuti=mpimg.imread(pathIm4 + "Ancuti3_RecoveredAncuti.png")
###########################################   Bazeille
img1_bazeille=mpimg.imread(pathIm1 + "Ancuti1_RecoveredBazeille.png")
img2_bazeille=mpimg.imread(pathIm2 + "Im1_RecoveredBazeille.jpg")
img3_bazeille=mpimg.imread(pathIm3 + "Eustice4_RecoveredBazeille.png")
img4_bazeille=mpimg.imread(pathIm4 + "Ancuti3_RecoveredBazeille.jpg")
###########################################   Carlevaris
img1_carlevaris=mpimg.imread(pathIm1 + "Ancuti1_RecoveredCarlevaris.png")
img2_carlevaris=mpimg.imread(pathIm2 + "Im1_RecoveredCarlevaris.png")
img3_carlevaris=mpimg.imread(pathIm3 + "Eustice4_RecoveredCarlevaris.png")
img4_carlevaris=mpimg.imread(pathIm4 + "Ancuti3_RecoveredCarlevaris.png")
###########################################   Chiang
img1_chiang=mpimg.imread(pathIm1 + "Ancuti1_RecoveredChiang.jpg")
img2_chiang=mpimg.imread(pathIm2 + "Im1_RecoveredChiang.jpg")
img3_chiang=mpimg.imread(pathIm3 + "Eustice4_RecoveredChiang.jpg")
img4_chiang=mpimg.imread(pathIm4 + "Ancuti3_RecoveredChiang.jpg")
###########################################   Lu
img1_TIF=mpimg.imread(pathIm1 + "Ancuti1_RecoveredTIF.jpg")
img2_TIF=mpimg.imread(pathIm2 + "Im1_RecoveredTIF.jpg")
img3_TIF=mpimg.imread(pathIm3 + "Eustice4_RecoveredTIF.jpg")
img4_TIF=mpimg.imread(pathIm4 + "Ancuti3_RecoveredTIF.jpg")


def plot_multi_bars_means_stds_sats(image_list, ax):
    """
    Given a list of RGB images and an ax within a fig, this plots 
    a three bar plot for each image; in the first bar it will draw the maximum distance
    between the mean_R, mean_G, mean_B; in the second bar, it will do the same
    for standard deviation, and in the third bar, will plot the inverse of the
    saturation of each image. This will plot the set of three bar plots one next
    to the other for as many images as there are in image_list
    Ticks are customized for a six images list
    Returns the mean of means, stds and sats for each image in a vector of 
    lenght equal to the lenght of image_list
    """    
    N = len(image_list)
    means = [compute_means(image) for image in image_list]
    stds = [compute_stds(image) for image in image_list]
    saturations = [(1-compute_saturations(image)) for image in image_list]
    
    mean_of_every_feature = []
    for idx in range(len(means)):
        mean_of_every_feature.append((means[idx] + stds[idx] + saturations[idx])/3)
        
    
    ## necessary variables
    ind = np.arange(N)                # the x locations for the groups
    width = 0.3                      # the width of the bars
    #the bars
    rects1 = ax.bar(ind, means, width, color='red')
    rects2 = ax.bar(ind+width, stds, width, color='green')
    rects3 = ax.bar(ind+2*width, saturations, width, color='blue')
    # axes and labels
    ax.set_xlim(-0.5*width,len(ind)+0.5*width)
    ax.set_ylim(0,1)# this is customized for optimal visualization
#    ax.set_xlabel(r'$Methods \ in$')
    
    #ax.set_title('Scores by group and gender')
    xTickMarks = [r'$[9]$', 
                  r'$[23]$', 
                  r'$[17]$', 
                  r'$[18]$', 
                  r'$[24]$', 
                  r'RC']
    ax.set_xticks(ind+width)
    xtickNames = ax.set_xticklabels(xTickMarks)
    plt.setp(xtickNames, rotation=0)
    ## add a legend
    ax.legend( (rects1[0], rects2[0], rects3[0]), (r'$\mu_{\mathrm{diff}}$', r'$\sigma_{\mathrm{diff}}$', r'$\lambda$'), 
              loc=1, ncol=3, handlelength=0.8, borderpad=0.2, labelspacing=0.0)

    return mean_of_every_feature
    
    
    
#Bar Plot with mean and std to show color cast and dominancy removal
plt.rcParams.update({'font.size': 24, 'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix'})


## the data
image1_methods = [img1_ancuti,img1_bazeille,img1_carlevaris,img1_chiang,img1_TIF,img1_me]
image2_methods = [img2_ancuti,img2_bazeille,img2_carlevaris,img2_chiang,img2_TIF,img2_me]
image3_methods = [img3_ancuti,img3_bazeille,img3_carlevaris,img3_chiang,img3_TIF,img3_me]
image4_methods = [img4_ancuti,img4_bazeille,img4_carlevaris,img4_chiang,img4_TIF,img4_me]

images_methods = [image1_methods, image2_methods, image3_methods, image4_methods]

for idx1 in range(len(images_methods)):
    for idx2 in range(len(images_methods[idx1])):
        if images_methods[idx1][idx2].max() > 1.0:
            images_methods[idx1][idx2]=images_methods[idx1][idx2].astype('float32')           
            images_methods[idx1][idx2]/=255.0
            



## the plots
fig, axes = plt.subplots(2,2,figsize=(19.5,9.5))

fig2, axes2 = plt.subplots(2,2,figsize=(19.5,9.5))


def simple_chart_plot(vector,ax):
    N = len(vector)
    ## necessary variables
    ind = np.arange(N)    # the x locations for the groups
    width = 1   
    ax.set_xlim(-0.5*width,len(ind)+0.5*width)    
    ax.set_ylim(0.2,0.3)  # this is customized for optimal visualization
    ax.bar(ind, vector, width, alpha=0.4, color=['blue', 'red', 'green','yellow', 'magenta', 'cyan'])
    xTickMarks = [r'$[9]$', 
                  r'$[23]$', 
                  r'$[17]$', 
                  r'$[18]$', 
                  r'$[24]$', 
                  r'Ours']
    ax.set_xticks(ind+0.5*width)
    xtickNames = ax.set_xticklabels(xTickMarks)
    plt.setp(xtickNames, rotation=0)
    
    
    
counter_for_images = 0
for i in range(2):
    for j in range(2):
        mean_of_all_i_this_image = plot_multi_bars_means_stds_sats(images_methods[counter_for_images], axes[i,j])
        print mean_of_all_i_this_image      
        simple_chart_plot(mean_of_all_i_this_image, axes2[i,j])
        
        

        axes[i,j].set_title('Image ' + str(counter_for_images + 1))
        axes[i,j].spines['top'].set_visible(False)
        axes[i,j].spines['left'].set_visible(False)
        axes[i,j].tick_params(top='off')
        
        
        axes2[i,j].set_title('Image ' + str(counter_for_images + 1))
        axes2[i,j].spines['top'].set_visible(False)
        axes2[i,j].spines['left'].set_visible(False)
        axes2[i,j].tick_params(top='off')
        
        counter_for_images += 1














plt.rcParams.update({'font.size': 24, 'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix'})
fig.tight_layout() 
fig.savefig("color_cast.pdf", dpi=1000,  transparent = True, edgecolor='none')
fig2.tight_layout() 
fig2.savefig("color_cast_stats.pdf", dpi=1000)        
#figManager = plt.get_current_fig_manager()
#figManager.window.showMaximized() 
        
        
