# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 18:45:27 2020

@author: Vrushali
"""

import cv2
import numpy as np
import os
from os.path import isfile, join
pathIn= r'Basketball\img\\'
bb = open('Basketball.txt','r').readlines()
pathOut = 'basketball_video_ms_bb.avi'
fps = 60

frame_array = []
files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
#for sorting the file names properly
files.sort(key = lambda x: x[5:-4])
files.sort()

filename=pathIn + files[0]
img1 = cv2.imread(filename)

#Siamese bb
box_coord_array = []
for i in bb:
    box_coord_array.append(i.strip())


#Meanshift bb
c,r,w,h = [int(round(float(i))) for i in box_coord_array[0].split(',')]
track_window = (c,r,w,h)
print(track_window)
ms_bb = [track_window]

# set up the ROI for tracking
roi = img1[r:r+h, c:c+w]
hsv_roi =  cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

#mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
mask = cv2.inRange(hsv_roi, np.array((0.)), np.array((180.)))
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

# Setup the termination criteria, either 10 iteration or move by at least 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )


for i in range(len(files)-1):
    filename=pathIn + files[i+1]
    print(".....",filename)
    frame = cv2.imread(filename)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
    
    track_win = box_coord_array[i+1].split(',')
    track_window = [int(round(float(win))) for win in track_win]
    
    # apply meanshift to get the new location
    ret, track_window = cv2.meanShift(dst, track_window, term_crit)    
    ms_bb.append(track_window)
    
    #with bounding box
    x,y,w,h = track_window
    #print(x,"\\",y,"\\",w,"\\",h)
    
    img2 = cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)
    cv2.imshow('img2',img2)

    cv2.waitKey(1)
    cv2.destroyAllWindows()
    height, width,layers = img2.shape
    size = (width,height)
    #print("size of basket",size)
    
    #inserting the frames into an image array
    frame_array.append(img2)


out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
for i in range(len(frame_array)):
    # writing to a image array
    out.write(frame_array[i])
print('Done!')
out.release()
cv2.destroyAllWindows()
