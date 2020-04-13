# Meanshift using Weighted centre of mass
# Input: correlation maps
# Output: text file with bounding box coordinates 

import torch
import cv2
import os
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
import torch.nn.functional as F
import math

# Get centre of mass
def calc_com(window):
    x_c = math.floor(window.shape[0]/2)
    y_c = math.floor(window.shape[1]/2)
    sxn = 0
    syn = 0
    sd = 0
    for i in range(window.shape[0]):
        for j in range(window.shape[1]):
            #numerator
            sxn += window[i,j]*(i-x_c)
            syn += window[i,j]*(j-y_c)
            #deno
            sd += window[i,j]
    
    comx = sxn/sd
    comy = syn/sd
    
    return comx,comy

# Correlation maps
corr_files = []
for i in range(724):
    path = 'corr_maps/bar_'+str(i)+'.pt'
    corr_files += [torch.load(path, map_location='cpu')]
print("Corr maps", len(corr_files))

# Images
pathIn= r'testdata\Basketball\img\\'
imgfiles = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
#for sorting the file names properly
imgfiles.sort(key = lambda x: x[5:-4])
imgfiles.sort()
imgfiles = imgfiles[:-1]  #725 imgs
print("Images",len(imgfiles))

# Groundtruth values
bb = open('testdata\Basketball\groundtruth_rect.txt','r').readlines()
bb_1 = bb[0].strip()
x,y,w,h = [int(i) for i in bb_1.split(',')]
print("Original", x,y,w,h)

cx, cy, cw, ch = 180, 171, 34, 81 #random initial bounding box on correlation maps


new_bb = []
for i in range(len(corr_files)):

    print("File:",i)
    corr_ = corr_files[i]
    # Upsample with a factor of 20
    p2 = F.interpolate(corr_, scale_factor=20, mode='bilinear', align_corners=False)
    arr_ = p2.detach().numpy()
    arr_ = np.squeeze(arr_)
    
    iter = 1
    while True:
        window = arr_[cx:cx+cw,cy:cy+ch]
        
        x_len = window.shape[0]
        y_len = window.shape[1]
        x_c = math.floor(x_len/2)
        y_c = math.floor(y_len/2)
        comx_, comy_ = calc_com(window)
        print("Centre of mass", comx_, comy_)
        if (abs(comx_ < 0.00001) and abs(comy_ < 0.00001)) or iter > 15:
            break
        else:
            deltax = math.ceil(comx_) if comx_>0 else math.floor(comx_)
            deltay = math.ceil(comy_) if comy_>0 else math.floor(comy_)
            cx += deltax
            cy += deltay
            
            x += math.ceil(deltax) if deltax>0 else math.floor(deltax)
            y += math.ceil(deltay) if deltay>0 else math.floor(deltay)
            iter += 1
        

    # Plot correlation maps
#    fig0,ax0 = plt.subplots(1)
#    ax0.imshow(arr_)
#    rect = patches.Rectangle((cx,cy),cw,ch,linewidth=1,edgecolor='r',facecolor='none')
#    ax0.add_patch(rect)
#    plt.show()
    
    
    # Plot original images
    img_ = pathIn + imgfiles[i]
    img_2 = cv2.imread(img_)
    img3 = cv2.rectangle(img_2, (x,y), (x+w,y+h), 255,2)
    cv2.imshow('img3',img3)
    cv2.waitKey(2)
    cv2.destroyAllWindows()
    
    new_bb += [[x,y,w,h]]
    
# Generate text file to record the coordinates of bounding boxes
newbb1 = np.array(new_bb, np.int32)
np.savetxt('coordinates\SiamMS_bb.txt', newbb1.astype(int), fmt='%i', delimiter=',', newline='\n')
    
    