# import the necessary packages
from collections import namedtuple
import numpy as np
import cv2
import os
from os.path import isfile, join
import matplotlib.pyplot as plt

# define the `Detection` object
Detection = namedtuple("Detection", ["image_path", "gt", "pred"])
fps = 50

def bb_intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0]+boxA[2], boxB[0]+boxB[2])
    yB = min(boxA[1]+boxA[3], boxB[1]+boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxA[2] * boxA[3]
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou,interArea

def get_bb_coords(bb):
    bb_gt = bb.strip()
    x,y,w,h = [float(i) for i in bb_gt.split(',')]
    return [x,y,w,h]

bb = open(r'groundtruth_rect.txt','r').readlines()
pred = open(r'test3.txt','r').readlines()
siam =open(r'Basketball.txt','r').readlines()
pathOut = 'tracking_ms.avi'

pathIn= r'C:\Users\18148\pysot\testing_dataset\OTB100\Basketball\img\\'
imgfiles = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
imgfiles.sort(key = lambda x: x[5:-4])
imgfiles.sort()
#imgfiles = imgfiles[:-1]  #725 imgs
i=0
frame_array=[]
iou_ms_array, overlap_ms_array=[],[]
iou_siam_array, overlap_siam_array=[],[]

# loop over the example detections

for img in imgfiles:
	# load the coordinates of the box of ground truth,siamese and meanshift
    bb_gt = get_bb_coords(bb[i])
    x,y,w,h = bb_gt
    x,y,w,h =int(x),int(y),int(w),int(h)
    bb_pred = get_bb_coords(pred[i])
    x1,y1,w1,h1 = bb_pred
    x1,y1,w1,h1  =int(x1),int(y1),int(w1),int(h1)
    bb_siam = get_bb_coords(siam[i])

    detection = Detection(join(pathIn, img),bb_gt,bb_pred)
    detection_siam = Detection(join(pathIn, img),bb_gt,bb_siam)

    image = cv2.imread(detection.image_path)
    #draw the ground-truth bounding box along with the predicted
	# bounding box
    cv2.rectangle(image, (x,y),(x+w,y+h), (0, 255, 0), 2)
    cv2.rectangle(image, (x1,y1),(x1+w1,y1+h1), (0, 0, 255), 2)
    # compute the intersection over union and display it
    iou,overlap = bb_intersection_over_union(detection.gt, detection.pred)
    iou_ms_array.append(iou)
    overlap_ms_array.append(overlap)
    
    iou_siam,overlap_siam = bb_intersection_over_union(detection_siam.gt, detection_siam.pred)
    iou_siam_array.append(iou_siam)
    overlap_siam_array.append(overlap_siam)

    cv2.putText(image, "IoU: {:.4f}".format(iou), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    #print("{}: {:.4f}".format(detection.image_path, iou))

    height, width,layers = image.shape
    size = (width,height)
    frame_array.append(image)
    #cv2.imshow("Image", image)
    cv2.waitKey(0)
    i+=1

out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
for i in range(len(frame_array)):
    # writing to a image array
    out.write(frame_array[i])
print('Done!')
out.release()
cv2.destroyAllWindows()

plt.plot(range(len(iou_ms_array)), iou_ms_array,range(len(iou_siam_array)), iou_siam_array, linewidth=1.0)
plt.show()

av_ms =np.mean(iou_ms_array)
av_siam  =np.mean(iou_siam_array)
print(av_ms,av_siam)




#print(iou_ms_array)
arr_ms=[]
arr_siam=[]
steps_29_frames_ms = np.reshape(iou_ms_array, (5, 145))
steps_29_frames_siam = np.reshape(iou_siam_array, (5, 145)) ###25,29



for i in range(1,6):
    arr_ms.append(np.mean(steps_29_frames_ms[i-1]))

for i in range(1,6):
    arr_siam.append(np.mean(steps_29_frames_siam[i-1]))


# for i in range(len(arr_ms)):
#     print(arr_ms[i],arr_siam[i])


iou_ms_array = np.asarray(iou_ms_array).astype('float')
iou_ms_array[iou_ms_array <0.3] = np.nan
#print(iou_ms_array)
means = np.nanmean(iou_ms_array)
print(means)


steps_29_frames_updated = np.reshape(iou_ms_array, (5, 145))
arr_ms_updated =[]
for i in range(1,6):
    arr_ms_updated.append(np.nanmean(steps_29_frames_updated[i-1]))

for i in range(len(arr_ms)):
    print(round(arr_ms[i],3),round(arr_ms_updated[i],3),round(arr_siam[i],3))


ao_len=[]
ao_len_siam=[]
#Expected Average Overlap
for i in [145,290,435,580,725]:
    ao =np.sum(overlap_ms_array[:i]) / i
    print(ao)
    ao_siam =np.sum(overlap_siam_array[:i])/i
    print(ao_siam)
    ao_len.append(ao)
    ao_len_siam.append(ao_siam)

print("Here")
den =725-145
print(ao_len)
print(ao_len_siam)
eao =np.sum(ao_len)/den
eao_siam =np.sum(ao_len_siam)/den

print(eao,eao_siam)

#Precision


def pre_rec_acc(arr):
    tp=0
    fp=0
    fn=0
    for i in range(len(arr)):
        if arr[i]>0 and arr[i]<0.5:
            fp+=1
        elif arr[i]>0.5:
            tp+=1
        else:
            fn+=1

    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    accuracy = tp/(tp+fp+fn)
    return precision,recall,accuracy

print(pre_rec_acc(iou_ms_array))
print(pre_rec_acc(iou_siam_array))


