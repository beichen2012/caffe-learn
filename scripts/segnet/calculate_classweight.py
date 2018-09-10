# coding: utf-8
import numpy as np
import cv2
import os
import pdb

weight = []
freqc = []
lists = []

root_path = '../original_data_wyj/111'
anno_path = os.path.join(root_path, 'trainannot')
width=448
height=800
class_num = 10 
img_names = []
for imgs in os.listdir(anno_path):
    img_names.append(imgs)
print('image num is: {}'.format(len(img_names))) 

for i in range(0,class_num):             #The length is decided by the number of classes in the image.
    # pdb.set_trace()
    count = 0
    sums = 0
    for item in img_names:
        img = cv2.imread(os.path.join(anno_path, item))
        #print('now process -> {}'.format(item))
        num = np.sum(img == i)
        if num != 0:
            count += 1
            sums += num
    if count != 0:
        freqc.append(sums/float(count*width*height))                #288*512 is the size of the image.
# print freqc
freq = np.median(freqc)
for i in range(len(freqc)):
    weight.append(freq/freqc[i])
print weight

print "ok"
