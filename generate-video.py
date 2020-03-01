import cv2
import numpy as np
import os
from os.path import isfile, join
pathIn= r'C:\Users\18148\pysot\testing_dataset\OTB100\Basketball\img\\'
pathOut = 'video_new.avi'
fps = 180
frame_array = []
files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]

#for sorting the file names properly
files.sort(key = lambda x: x[5:-4])
files.sort()
frame_array = []
files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]

# setup initial location of window
# r,h,c,w - region of image
#           simply hardcoded the values
r,h,c,w =214,81,198,34
track_window = (c,r,w,h)
frame = cv2.imread(r'C:\Users\18148\pysot\testing_dataset\OTB100\Basketball\img\0001.jpg')

# set up the ROI for tracking
roi = frame[r:r+h, c:c+w]
hsv_roi =  cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

# Setup the termination criteria, either 10 iteration or move by at least 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

#for sorting the file names properly
files.sort(key = lambda x: x[5:-4])

#for reading the outputted coordinates of siamrpn
pathBasket = r'C:\Users\18148\pysot\results\OTB100\experiments\siamrpn_alex_dwxcorr\model\Basketball.txt'
i=0
with open(pathBasket) as file_in:
    for line in file_in:
        filename=pathIn + files[i]
        i+=1
        #reading each files or read the frame
        frame = cv2.imread(filename)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

        #for mean-shift
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)


        # Draw it on image
        x,y,w,h = line.split(',')
        x,y,w,h =int(float(x)),int(float(y)),int(float(w)),int(float(h))
        
        # Start coordinate, here (100, 50) 
        # represents the top left corner of rectangle 
        start_point = (x,y)

        # Ending coordinate, here (125, 80) 
        # represents the bottom right corner of rectangle 
        end_point = ((x+w),(y+h)) 

        # Black color in BGR 
        color = (255, 0, 0) 

        # Line thickness of -1 px 
        # Thickness of -1 will fill the entire shape 
        thickness = 2

        img2 = cv2.rectangle(frame, start_point, end_point, color ,thickness)
        cv2.imshow('img2',img2)
    
        cv2.waitKey(60)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        height, width,layers = img2.shape
        size = (width,height)
        print("size of basket",size)
    
        #inserting the frames into an image array
        frame_array.append(img2)



out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
for i in range(len(frame_array)):
    # writing to a image array
    out.write(frame_array[i])
print('Done!')
out.release()
cv2.destroyAllWindows()
