#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 12:08:08 2017
#!!!! mean this line is temporarily marked
@author: user
"""
from PIL import Image
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
plt.rcParams['figure.figsize'] = (10, 10)
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
#path ='/home/'+user_name+'/SY/classfy_photos/'+region
#path = '/home/pan/SY/flickr_project/BBox-Label-Tool-master/Images/111'

#====== count number of class =========        

user_name = 'pan' #for difference computer
region = '111'
class_set = set()
[dataset,num_img,img_format] =  util4pan.read_dataset(path = '/home/'+user_name+'/SY/',file_name = region+'_dataset.txt')
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
#path4photo = '/home/pan/SY/flickr_project/BBox-Label-Tool-master/Images/test4BBOX' #longSangTemple
path4photo = '/home/pan/SY/flickr_project/BBox-Label-Tool-master/Images/'+region #!!!! need change, after complete
inputs = []
images = []
photos_list = [f_name for f_name in os.listdir(path4photo) if not os.path.isdir(path4photo+'/'+ f_name)]

#Image.fromarray(images[1]) photos_list[1]
for photo_name in photos_list:
    img_path = path4photo+'/'+photo_name
    img = image.load_img(img_path, target_size=(300, 300))
    img = image.img_to_array(img)
    images.append(imread(img_path))#only get pixel info. [height,weight,3] for each image
    inputs.append(img.copy())
  
inputs = preprocess_input(np.array(inputs))#只是rgb各扣掉1個常數（Zero-center by mean pixel）

#========= predict ======================
preds = model.predict(inputs, batch_size=1, verbose=2)
results = bbox_util.detection_out(preds)


#========== Visualization =====================
crop_image = {} #format photoID:[[photiID,label1,BBOX1],[photiID,label2,BBOX2],...]
'''
for i,name in enumerate(photos_list):
    Image.open(path4photo+'/'+name).save("/home/pan/Desktop/ssd_keras-master/bbox_result/"+str(i)+".jpg")#get image
'''
    
bbox_list = [] #for store
count =0
for i, img in enumerate(images):      
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
    #print("name:{}".format(photos_list[i]))# show photo name
    img_crop_list =[]
    Img = Image.open(path4photo+'/'+photos_list[i])#get image   
    #Img.show()
    #plt.show()
    for j in range(top_conf.shape[0]):
        xmin = int(round(top_xmin[j] * img.shape[1]))
        ymin = int(round(top_ymin[j] * img.shape[0]))
        xmax = int(round(top_xmax[j] * img.shape[1]))
        ymax = int(round(top_ymax[j] * img.shape[0]))
        score = top_conf[j]
        label = int(top_label_indices[j])
        label_name = my_classes[label - 1] #due to '0' be assign to background
        display_txt = '{:0.2f}, {}'.format(score, label_name)
        coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
        color = colors[label]
        currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
        currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor':color, 'alpha':0.5})
        #===
        bbox_list.append([photos_list[i],label_name,xmin,ymin,xmax,ymax])   #!!!!!!!!!!!!!!!!!!!!![photo_id,label,BBOX]
        print("BOX:[{} {} {} {}]\n".format(xmin,ymin,xmax,ymax))# show photo name    
            #img_crop_list format : [[photo_id,label,img ],[photo_id,label,img ],...  ]
        img_crop_list.append([photos_list[i],label_name,Img.crop((xmin,ymin,xmax,ymax)).resize( (300, 300), Image.NEAREST )])#!!!!!!!!!!!!!!!!!!!!!!!!
        Img.crop((xmin,ymin,xmax,ymax)).save('/home/pan/Desktop/ssd_keras-master/bbox/'+str(count)+'.jpg')
        count +=1
            #Img.crop((xmin,ymin,xmax,ymax)).show()
            #plt.imshow(np.array(Img.crop((xmin,ymin,xmax,ymax))) / 255.)
            #plt.show()
    plt.savefig("/home/pan/Desktop/ssd_keras-master/bbox_result/"+str(i)+".jpg")#save photo with BBOX
    plt.show()#!!!!!!!!!!!!!!!!!!!!!!!!
    photos_list[:10]
        #=============     
                                                # np.array(img.crop((xmin,ymin,xmax,ymax)) #get BBOX region
    if len(img_crop_list):#if at least one BBOX     #!!!!
        crop_image[photos_list[i]] = img_crop_list #store  #!!!!

#=== show BBOX

bbox_img = 2
'''
photo_name = list(crop_image.keys())[2]#len(list(crop_image.keys())) len(photos_list)
for box in crop_image[photo_name]:#list(crop_image.keys())[1]
    print("Id:{} label:{}".format(box[0],box[1]))
    box[bbox_img].show()
'''    


# === get features from last layer of VGG16 and apply t-SNE========================

SSD_4096 = Model(inputs=model.input, outputs=model.get_layer('fc7').output)
#vgg_4096 = Model(inputs=model.input, outputs=model.get_layer('fc2').output)
#np.shape(SSD_4096)
#img_path = '/home/pan/ssd_keras-master/test96'
#img = image.load_img(img_path, target_size=(300, 300))
#np.shape(img)

features = []
labeles4feature = []
save_label =[]
save_label = util4pan.open_list_file(flie_name = './bbox_label_list'+region+'.txt')
#save_label[-5:]
#labeles4feature[-5:]
count =0
for i,B in enumerate(sorted(crop_image.keys())):    #crop_image[list(crop_image.keys())[0]]
    #if i<10:# only try 10 example
    print("[{}] name:{}".format(i,B))
    for j,box_ in enumerate(crop_image[B]):        
        print("count: ",count)
        plt.imshow(np.array(box_[bbox_img]) / 255.)
        plt.show()#show BBOX as photo
        #==== manually add label ====
        #label = input("input label:")        
        label = save_label[count]
        crop_image[B][j].append(label)#!!!!
        save_label.append(label)
        #======
        img = np.expand_dims(np.array(box_[bbox_img]), axis=0)
        feature = SSD_4096.predict(img)#get feature
        feature = np.reshape(feature,(1,-1))
        features.append(feature)   
        labeles4feature.append(label)#1:label name
        count += 1
    #else:
    #    break
print("label finished!!!\n")
#===== temporarily store label info ============
#util4pan.save_list_file(save_label,flie_name = '/home/pan/Desktop/ssd_keras-master/bbox_label_list.txt',path = './')
#util4pan.save_list_file(list(crop_image.keys()),flie_name = 'readImageOrder.txt',path = './')

'''
#==== show manually label result ====
for i,B in enumerate(crop_image):   
    if      i>10:
        break
    for j,box_ in enumerate(crop_image[B]):             
        plt.imshow(np.array(box_[bbox_img]) / 255.)
        plt.show()
        labeles4feature.append(crop_image[B][j][-1])#1:label name
        print("label:",labeles4feature[-1])
'''
#== transfer feature to nd array ============
if len(features) <2:
    print("lenth of features < 2,it will cause error in t-SNE!!!!\n")
#np.shape(features_arr)
features_arr = features[0]
for i in range(1,len(features)):
    features_arr = np.concatenate([features_arr ,features[i]],axis=0)

# ====== t-SNE ==============================
from sklearn.manifold import TSNE
print("ploting TSNE result(it take minutes)..............\n")
features2D = TSNE(n_components=2,random_state=20170818).fit_transform(features_arr)#np.shape(np.reshape(x,(224,-1)))
#plot in 2D
for i in range(np.shape(features2D)[0]):
    if labeles4feature[i] == '1':
        plt.plot(features2D[i,0],features2D[i,1],'b*')#101
    elif labeles4feature[i] == '2':
        plt.plot(features2D[i,0],features2D[i,1],'g*')#taipei hotel
    elif labeles4feature[i] == '3':
        plt.plot(features2D[i,0],features2D[i,1],'r*')# red house
    elif labeles4feature[i] == '4':
        plt.plot(features2D[i,0],features2D[i,1],'*',color = '#C8FD3D')#great hotel
    elif labeles4feature[i] == '0':
        plt.plot(features2D[i,0],features2D[i,1],'*',color = '#E11DE7')#pink
plt.savefig("t-SNE_vgg16-"+str(int(time.time()))+".jpg")
    #len(features2D[:,0])    
#====== color and style ===========
#plt.plot(1,1,'s',color = '#76FDA8')# light green
#plt.plot(2,1,'s',color = '#06C2AC')# dark green
#plt.plot(1,1.5,'s',color = '#F36196')#E11DE7' 
# '#C8FD3D' = yello green
# '#E11DE7' = pink    
#==================================
'''    
index =0
for i,B in enumerate(crop_image):    
    if i<10:# only try 10 example
        print("name:{}".format(B))
        for j,box_ in enumerate(crop_image[B]):
            #plt.imshow(np.array(box_[bbox_img]) / 255.)
            #plt.show()                     
            features.append(feature)
         #   crop_image[B][j].append(features2D[index]) #store 2D-feature in corresponding BBOX    
            index+=1
    else:
        break
'''
#print("np.shape(feature):",np.shape(features))

#a = np.reshape(feature,(1,-1))#np.shape(a) 
#features = a.copy()
#features = np.concatenate([features ,a],axis=0)  #[]
#c = np.concatenate([b ,a],axis=0)  #[]
#np.shape(c)

#======== test ====================
'''
img4vgg = crop_image[photo_name][0]
features = []
from keras.applications.vgg19 import VGG19
#from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
#from keras.models import Model
#import numpy as np

base_model = VGG19(weights='imagenet')
modelVgg19_part = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)
#img_path = '/home/pan/ssd_keras-master/test96'
#img = image.load_img(img_path, target_size=(224, 224))
for key in crop_image:
    for box in crop_image[key]:
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features.append(modelVgg19_part.predict(x))
        #np.shape(feature)#np.shape(x)
'''
# ======= t-SNE =============================
'''
from sklearn.manifold import TSNE
#X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
#X_embedded = TSNE(n_components=2).fit_transform(feature)
y = np.array([1,1,1])#np.shape(y)
bbox_2D = TSNE(n_components=2).fit_transform(y) #np.shape(np.reshape(x,(224,-1)))
plt.plot(bbox_2D[:,0],bbox_2D[:,1],'r*')#X_embedded
#X_embedded.shape
'''

'''   
#======= save BBOX info (not complete)========================
def save_list_file(list_,flie_name = 'tags_list',path = '/home/user/'):    
    os.chdir(path)# change current path
    if not flie_name[-3:] == 'txt':
        print('remenber to add .txt in tail')
    with open(flie_name, 'w', encoding = 'UTF-8')  as f:     # 也可使用指定路徑等方式，如： C:\A.txt
        for s in list_:
            f.write(s[0]+'='+s[1]+'='+str(s[2])+'='+str(s[3])+'='+str(s[4])+'='+str(s[5])+'='+'\n')                 
save_list_file(bbox_list,flie_name = 'bbox4TestResult-'+my_classes[0],path = '/home/pan/SY/')
'''