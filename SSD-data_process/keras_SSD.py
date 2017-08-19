#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 12:08:08 2017

@author: user
"""

import cv2
import keras
from keras.applications.imagenet_utils import preprocess_input
from keras.backend.tensorflow_backend import set_session
from keras.models import Model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imread
import tensorflow as tf

from ssd import SSD300
from ssd_utils import BBoxUtility
import os
import util4pan
import time
#%matplotlib inline
plt.rcParams['figure.figsize'] = (8, 8)
plt.rcParams['image.interpolation'] = 'nearest'

np.set_printoptions(suppress=True)

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.45
set_session(tf.Session(config=config))
'''
voc_classes = ['Aeroplane', 'Bicycle', 'Bird', 'Boat', 'Bottle',
               'Bus', 'Car', 'Cat', 'Chair', 'Cow', 'Diningtable',
               'Dog', 'Horse','Motorbike', 'Person', 'Pottedplant',
               'Sheep', 'Sofa', 'Train', 'Tvmonitor']
'''
#my_classes = ['cksHall', 'concertHall', 'theaterHall', 'taipei101']

#====== count number of class =========        

user_name = 'pan' #for difference computer
region = 'voc+112'
class_set = set()
[dataset,num_img,img_format] =  util4pan.read_dataset(path = '/home/'+user_name+'/SY',file_name = region+'_dataset.txt')
#path4image = '/home/'+user_name+'/python_download_image/VOCdevkit/VOC2012/voc+112'  
#path4bbox = '/home/'+user_name+'/python_download_image/VOCdevkit/VOC2012/Annotations'
'''
i = 2
while i < len(dataset):
    box_num = int(lines[i])
    if box_num > 0:
        for j in range(1,box_num+1):
            class_set.add(lines[i+j].split('=')[2])#add label
    i += box_num+1
'''        
for img_id in dataset:
    dataset[img_id][2] # 2 is label
    class_set.add(dataset[img_id][2])
    
num_class = len(class_set)    
#============== make training data ===============  !!!note:class_set has no order and label2prob's order is by dictionary
# make one hot bector
label2prob = {}
prob2label = {}
prob = np.eye(num_class)
for i,class_name in enumerate(class_set):
    label2prob[class_name]=list(prob[i])
    prob2label[np.argmax(prob[i])] = class_name    
#==================================================

my_classes = prob2label
NUM_CLASSES = len(my_classes) + 1
     
#===============================                
#===============================            
input_shape=(300, 300, 3)
model = SSD300(input_shape, num_classes=NUM_CLASSES)
#model.load_weights('weights_SSD300.hdf5', by_name=True) 
#model.load_weights('./checkpoints/weights.29-1.78.hdf5', by_name=True)
model.load_weights('./checkpoints/voc12+building(SGD)-2017.8.18/weights.28-2.88.hdf5', by_name=True)


bbox_util = BBoxUtility(NUM_CLASSES)

#===============================
#path4photo ='./pics/sysHall'
#path4photo = '/home/pan/python_download_image/0000GPS_photo0000/0000待整理的標籤0000/中山'
#path4photo = '/home/pan/python_download_image/0000GPS_photo0000/0000待整理的標籤0000//Taipei 101 Skyscraper'
path4photo = '/home/pan/SY/flickr_project/BBox-Label-Tool-master/Images/111' #longSangTemple
#path4photo = '/home/pan/python_download_image/VOCdevkit/JPEGImages'
inputs = []
images = []
photos_list = [f_name for f_name in os.listdir(path4photo) if not os.path.isdir(path4photo+'/'+ f_name)]



for photo_name in photos_list:
    img_path = path4photo+'/'+photo_name
    img = image.load_img(img_path, target_size=(300, 300))
    img = image.img_to_array(img)
    images.append(imread(img_path))#only get pixel info. [height,weight,3] for each image
    inputs.append(img.copy())
  
'''    
img_path = './pics/cat.jpg'
img = image.load_img(img_path, target_size=(300, 300))
img = image.img_to_array(img)
images.append(imread(img_path))
inputs.append(img.copy())

img_path = './pics/15368725526=DSC02129.jpg=______101=25.031319=121.583577.jpg'
img = image.load_img(img_path, target_size=(300, 300))
img = image.img_to_array(img)
images.append(imread(img_path))
inputs.append(img.copy())
img_path = './pics/15979756904=Taipei 101 @ ______=______101=25.027366=121.576141.jpg'
img = image.load_img(img_path, target_size=(300, 300))
img = image.img_to_array(img)
images.append(imread(img_path))
inputs.append(img.copy())
img_path = './pics/3310110280=中山碑林=25.040071=121.559332.jpg'
img = image.load_img(img_path, target_size=(300, 300))
img = image.img_to_array(img)
images.append(imread(img_path))
inputs.append(img.copy())
'''
inputs = preprocess_input(np.array(inputs))#只是rgb各扣掉1個常數（Zero-center by mean pixel）

#===============================
preds = model.predict(inputs, batch_size=1, verbose=2)
results = bbox_util.detection_out(preds)
#np.shape(preds)
#len(results)
#===============================
# test
'''
a = model.predict(inputs, batch_size=1)
b = bbox_util.detection_out(preds)
np.shape(a)
'''

#===============================
bbox_list = [] #for store
for i, img in enumerate(images):
    if i>100:
        break
    # Parse the outputs.
    if results[i]!=[]:
        det_label = results[i][:, 0]
        det_conf = results[i][:, 1]
        det_xmin = results[i][:, 2]
        det_ymin = results[i][:, 3]
        det_xmax = results[i][:, 4]
        det_ymax = results[i][:, 5] # det_conf = results[0][:, 1] i=0
    else:#avoid []
        det_label = np.array([0.])
        det_conf = np.array([0.])
        det_xmin = np.array([0.])
        det_ymin = np.array([0.])
        det_xmax = np.array([0.])
        det_ymax = np.array([0.])
    # Get detections with confidence higher than 0.6.
    top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.6]
    
    top_conf = det_conf[top_indices]
    top_label_indices = det_label[top_indices].tolist()
    top_xmin = det_xmin[top_indices]
    top_ymin = det_ymin[top_indices]
    top_xmax = det_xmax[top_indices]
    top_ymax = det_ymax[top_indices]

    colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()

    plt.imshow(img / 255.)
    currentAxis = plt.gca()
    
    for i in range(top_conf.shape[0]):
        xmin = int(round(top_xmin[i] * img.shape[1]))
        ymin = int(round(top_ymin[i] * img.shape[0]))
        xmax = int(round(top_xmax[i] * img.shape[1]))
        ymax = int(round(top_ymax[i] * img.shape[0]))
        score = top_conf[i]
        label = int(top_label_indices[i])
        label_name = my_classes[label - 1]
        display_txt = '{:0.2f}, {}'.format(score, label_name)
        coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
        color = colors[label]
        currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
        currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor':color, 'alpha':0.5})
        #===
        bbox_list.append([photos_list[i],label_name,xmin,ymin,xmax,ymax])   
        
    plt.show()


#======= save BBOX info (not complete)========================
def save_list_file(list_,flie_name = 'tags_list',path = '/home/user/'):    
    os.chdir(path)# change current path
    if not flie_name[-3:] == 'txt':
        print('remenber to add .txt in tail')
    with open(flie_name, 'w', encoding = 'UTF-8')  as f:     # 也可使用指定路徑等方式，如： C:\A.txt
        for s in list_:
            f.write(s[0]+'='+s[1]+'='+str(s[2])+'='+str(s[3])+'='+str(s[4])+'='+str(s[5])+'='+'\n')                 
save_list_file(bbox_list,flie_name = 'bbox4TestResult-'+my_classes[0],path = '/home/pan/SY/')
