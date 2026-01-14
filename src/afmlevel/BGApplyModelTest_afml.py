# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 19:47:27 2025

@author: pymte

Testing the applymodel_bg function
"""


import afmlevel_functions as afml
import os
import numpy as np
from PIL import Image
from sklearn.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import cv2
from pytorch_msssim import MS_SSIM
import torch


def znorm(image):
    mean = np.mean(image)
    std_dev = np.std(image)
    imnorm_z = (image - mean) / std_dev
    return imnorm_z

def minmaxnorm(image):    
    if isinstance(image, torch.Tensor):
        im0 = image - image.min()
        imnorm = im0 / im0.max()
        return imnorm
    elif isinstance(image, np.ndarray):
        im0 = image - np.min(image)
        imnorm = im0 / np.max(im0)
        return imnorm

def xyplanefit(imarray, polyx, polyy):
     mc = np.mean(imarray, axis = 0) 
     x = np.arange(0, len(mc), 1) 
     p = np.polyfit(x, mc, polyx)  
     p = np.poly1d(p)          
     pvals = np.polyval(p, x)  
     r = imarray - pvals[np.newaxis, :] 

     mr = np.mean(r, axis = 1) 
     y = np.arange(0, len(mr), 1) 
     p = np.polyfit(y, mr, polyy)  
     p = np.poly1d(p)  
     pvals = np.polyval(p, y) 
     r = r - pvals[:, np.newaxis] 

     return r

def MS_SSIMloss(predicted, groundtruth, device='cpu'):
    if not isinstance(predicted, torch.Tensor):
        predicted = torch.from_numpy(predicted)
    if not isinstance(groundtruth, torch.Tensor):
        groundtruth = torch.from_numpy(groundtruth)

    predicted = predicted.float().to(device)
    groundtruth = groundtruth.float().to(device)
    predicted = predicted.unsqueeze(0).unsqueeze(0)
    groundtruth = groundtruth.unsqueeze(0).unsqueeze(0)
    
    MS_SSIMfn = MS_SSIM(data_range=1, size_average=True, channel=1)
    loss = 1 - MS_SSIMfn(predicted, groundtruth)
    return loss.item()


 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
saveimgs = False
label = 'BGfunctionpixsplit_apply3x_Alltestdata'

base_path = r'C:\Users\pymte\OneDrive - University of Leeds\_AFM Software\Level ML data'    
test_filename =  'All Test Data Nov25' #'Test Data from 3 and 4' #'Extra Test data Sep 2025'
test_path = os.path.join(base_path, test_filename) 

test_data = []
imageidlist = []
datatoexclude = ['imaging-15_02_32Image']
for filename in os.listdir(test_path):
    if filename.endswith('.tiff') and not filename.endswith('_levelled.tiff'):
        image_id = filename.split('.tiff')[0]
        if image_id in datatoexclude:
            continue
        imageidlist.append(image_id)

AFM = np.load('lutAFM.npy')
AFM = ListedColormap(AFM)

MSEscore = False
SSIMscore = True
PSNRscore = True
saveimgs = False


if saveimgs:
    image_save_dir = f'{label}_images'
    os.makedirs(image_save_dir, exist_ok=True)

MSElist = []
SSIMlist = []
PSNRlist = []

i = 0
for image_id in imageidlist:
    image_file = f'{image_id}.tiff'
    levelled_file = f'{image_id}_levelled.tiff'
    image_path = os.path.join(test_path, image_file) 
    levelled_path = os.path.join(test_path, levelled_file) 
    
    image = Image.open(image_path) #open image    
    imarray = np.array(image)  #convert to array                 
    
    levelled = Image.open(levelled_path) #open levelled image
    #levelled = levelled.resize([256,256], Image.NEAREST) #RESIZE TEST. REMOVE IF NOT NEEDED
    levarray = np.array(levelled)     #convert to array                       
    levnorm = znorm(levarray)
           
    model_bg, model_levarray = afml.applymodel_bg_pixelsplit_all(imarray, 3)
    model_levnorm = znorm(model_levarray)
    levnorm = levnorm.astype(np.float32)
    model_levnorm = model_levnorm.astype(np.float32)
    
    model_bg2, model_levarray2 = afml.applymodel_bg_pixelsplit_all(model_levarray, 3) #second application of BG model
    model_levnorm2 = znorm(model_levarray2)
    
    model_bg3, model_levarray3 = afml.applymodel_bg_pixelsplit_all(model_levarray2, 3) #second application of BG model
    model_levnorm3 = znorm(model_levarray3)
    #model_levnorm = cv2.resize(model_levnorm, [256,256], interpolation=cv2.INTER_NEAREST) #RESIZE TEST. REMOVE IF NOT NEEDED
    
    if MSEscore:
        MSE = mean_squared_error(model_levnorm, levnorm)    
        MSE2 = mean_squared_error(model_levnorm2, levnorm)
        MSE3 = mean_squared_error(model_levnorm3, levnorm)
        
        MSElist.append({'Image_ID': image_id, 'MSE': MSE, 'MSE2': MSE2, 'MSE3': MSE3})
    
    if SSIMscore or PSNRscore:
        levnorm_mm = minmaxnorm(levarray)
        model_levnorm_mm = minmaxnorm(model_levarray)
        model_levnorm2_mm = minmaxnorm(model_levarray2)
        model_levnorm3_mm = minmaxnorm(model_levarray3)
    
    if SSIMscore:
        MSSSIM = MS_SSIMloss(model_levnorm_mm, levnorm_mm)
        MSSSIM2 = MS_SSIMloss(model_levnorm2_mm, levnorm_mm)
        MSSSIM3 = MS_SSIMloss(model_levnorm3_mm, levnorm_mm)
        
        SSIMlist.append({'Image_ID': image_id, 'MSSSIM': MSSSIM, 'MSSSIM2': MSSSIM2, 'MSSSIM3': MSSSIM3})
        print(f'Image {i+1} MS SSIM: {MSSSIM:.2f}')
     
    if PSNRscore:
        PSNR = peak_signal_noise_ratio(model_levnorm_mm, levnorm_mm, data_range = 1) 
        PSNR2 = peak_signal_noise_ratio(model_levnorm2_mm, levnorm_mm, data_range = 1)
        PSNR3 = peak_signal_noise_ratio(model_levnorm3_mm, levnorm_mm, data_range = 1)
        
        PSNRlist.append({'Image_ID': image_id, 'PSNR': PSNR, 'PSNR2': PSNR2, 'PSNR3': PSNR3})
        print(f'Image {i+1} PSNR: {PSNR:.2f}')
        
    if saveimgs:
        plt.figure(figsize=(24,4))
        plt.subplot(1,6,1)
        plt.title(f'image {i+1}')
        plt.imshow(imarray, cmap=AFM)  
        plt.colorbar(shrink = 0.7)
    
        plt.subplot(1,6,2)
        plt.title("GT levelled")
        plt.imshow(levnorm, cmap=AFM, vmin=-5, vmax = 5) #choose to display un normalised levarray, or levnorm
        plt.colorbar(shrink = 0.7)
    
        plt.subplot(1,6,3)
        plt.title("BG model background")
        plt.imshow(model_bg, cmap=AFM)
        plt.colorbar(shrink = 0.7)
        
        plt.subplot(1,6,4)
        plt.title(f"BG model levelled - MSE:{MSE:.2f}")
        plt.imshow(model_levnorm, cmap=AFM, vmin=-5, vmax = 5) #choose to display un normalised levarray, or levnorm
        plt.colorbar(shrink = 0.7)
        
        #plt.subplot(1,8,5)
        #plt.title("BG model background 2")
        #plt.imshow(model_bg2, cmap=AFM)
        #plt.colorbar(shrink = 0.7)
        
        plt.subplot(1,6,5)
        plt.title(f"BG model levelled 2 - MSE:{MSE2:.2f}")
        plt.imshow(model_levnorm2, cmap=AFM, vmin=-5, vmax = 5) #choose to display un normalised levarray, or levnorm
        plt.colorbar(shrink = 0.7)
        
        #plt.subplot(1,8,7)
        #plt.title("BG model background 3")
        #plt.imshow(model_bg3, cmap=AFM)
        #plt.colorbar(shrink = 0.7)
        
        plt.subplot(1,6,6)
        plt.title(f"BG model levelled 3 - MSE:{MSE3:.2f}")
        plt.imshow(model_levnorm3, cmap=AFM, vmin=-5, vmax = 5) #choose to display un normalised levarray, or levnorm
        plt.colorbar(shrink = 0.7)
           
        save_path = os.path.join(image_save_dir, f'image_{i+1}.png')
        plt.savefig(save_path, bbox_inches='tight')
    
        plt.show()
    

    i += 1
 