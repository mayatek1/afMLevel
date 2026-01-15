# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 19:47:27 2025

@author: pymte

"""


import afmlevel_functions as afml
import os
import numpy as np
from PIL import Image
from sklearn.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import torch
AFM = np.load('lutAFM.npy')
AFM = ListedColormap(AFM)
  
# Apply background model and mask model to a chosen AFM image from the demo image folder
   
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
testimage = 3 #choose value from 0 to 20 to pick an image from the image folder
BGrpt = 1 #choose value from 1 to 3 to apply the BG model that number of times
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

image_path = r"C:\Users\pymte\OneDrive - University of Leeds\Project 2\Machine Learning\afMLevel\test_image_tiffs"    
images = []
for file in os.listdir(image_path):
    images.append(file)
    
image_path = os.path.join(image_path, images[testimage])    
image = Image.open(image_path) #open image    
imarray = np.array(image)  #convert to array  

plt.imshow(imarray, cmap = AFM) 
              
 
if BGrpt == 1 or BGrpt == 2 or BGrpt == 3:
    model_bg, model_levarray = afml.applymodel_bg(imarray, 3)    
    
if BGrpt == 2 or BGrpt == 3:
    model_bg2, model_levarray2 = afml.applymodel_bg(model_levarray, 3) #second application of BG model

if BGrpt == 3:
    model_bg3, model_levarray3 = afml.applymodel_bg(model_levarray2, 3) #third application of BG model

mask = afml.applymodel_mask(imarray)

if BGrpt == 1:
    BGlev =model_levarray   
if BGrpt == 2:
    BGlev =model_levarray2   
if BGrpt == 3:
    BGlev =model_levarray3   

plt.figure(figsize=(16,4))

plt.subplot(1,3,1)
plt.title("BG model background")
plt.imshow(model_bg, cmap=AFM)

plt.subplot(1,3,2)
plt.title("BG model levelled")
plt.imshow(BGlev, cmap = AFM) 

plt.subplot(1,3,3)
plt.title("Mask model mask")
plt.imshow(mask) 

plt.show()
