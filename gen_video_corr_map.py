import cv2
import os
from os.path import isfile, join

pathIn = r'C:\Users\18148\Desktop\eval_track\corrMap\\'
video_name = 'corr.avi'


images = []
for i in range(1,36):
    images.append('corr_'+str(i)+'.png')

print(images)
frame = cv2.imread(os.path.join(pathIn, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 10, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(pathIn, image)))

cv2.destroyAllWindows()
video.release()