# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 22:10:30 2020

@author: Vrushali
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import ndimage, misc
import torch.nn.functional as F
import math

    
x = torch.load('bar_0.pt', map_location='cpu')
arr2 = x.detach().numpy()
print(arr2.shape)
arr_2 = np.squeeze(arr2)
print(arr_2.shape)
plt.imshow(arr_2)
plt.show()

#Corr map 
arr = x.detach().numpy()
print(arr.shape)
arr_ = np.squeeze(arr)
print(arr_.shape)


# Upsampling
corr_map_size =125
upsc_size = (corr_map_size-1)*4 + 1
print("UPSC:",upsc_size)
p2 = F.interpolate(x, upsc_size, mode='bilinear', align_corners=False)
arr2 = p2.detach().numpy()
print(arr2.shape)
arr_2 = np.squeeze(arr2)
print(arr_2.shape)
plt.imshow(arr_2)
plt.show()


x_cord = 270
y_cord = 170
deltax = 1
deltay = 1
no=1
old =0
prev=0
frame_array = []
pathOut = 'corr_map.avi'
while True:

    window = arr_2[x_cord:x_cord+34,y_cord:y_cord+81]
    
    x_len = window.shape[0]
    y_len = window.shape[1]
    x_c = math.floor(x_len/2)
    y_c = math.floor(y_len/2)
    
    fig,ax = plt.subplots(1)
    ax.imshow(arr_2)

    rect = patches.Rectangle((x_cord,y_cord),34,81,linewidth=1,edgecolor='r',facecolor='none')
    ax.add_patch(rect)
    frame_array.append(rect)
    #plt.show()
    

    sxn = 0
    syn = 0
    sd = 0 

  
    if abs(window[x_c, y_c]-old)==prev:
        break
    
    #Shift
    
    for i in range(x_len):
        for j in range(y_len):
            #numerator
            sxn += window[i,j]*(i-x_c)
            syn += window[i,j]*(j-y_c)
            
            #deno
            sd += window[i,j]
            


    deltax = math.ceil(sxn/sd) if sxn>0 else math.floor(sxn/sd)
    deltay = math.ceil(syn/sd) if syn>0 else math.floor(syn/sd)

    x_cord += deltax
    y_cord += deltay
    
    no +=1
    prev = abs(window[x_c,y_c]-old)
    old = window[x_c,y_c]
    

out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
for i in range(len(frame_array)):
    # writing to a image array
    out.write(frame_array[i])
print('Done!')
out.release()
cv2.destroyAllWindows()


      
    