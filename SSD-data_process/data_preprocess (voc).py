#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 22:08:47 2017
@author: user
descript:classify image by GPS,and rename to add label to their name.
"""
import os
import numpy as np
import shutil
#import manipulate_file

max_lat = 26.
max_lon = 122.
lat = [0,25.024439,25.034215,25.042341,25.052082,25.061537,25.070958,max_lat]
lon = [0,121.495034,121.510641,121.527292,121.543331,121.559898,121.575844,max_lon]
lat = np.asarray(lat)
lon = np.asarray(lon)
homePath = os.getcwd()
#=========================================================
#open .txt and read each line, then store in list
#=========================================================
def open_list_file(flie_name):
    if not os.path.exists(flie_name):
        print('no sush file name(need include path)!')
        return
    with open(flie_name, 'r',encoding = 'UTF-8') as f: 
         lines = f.readlines()
         #print(lines)
    for i in range(len(lines)):
        lines[i] = lines[i][:-1]
     #   print(lines)
    return lines


#=========================================================
#save .txt and then store in file
#=========================================================
def save_list_file(list_,flie_name = 'tags_list',path = '/home/user/'):    
    os.chdir(path)# change current path
    if not flie_name[-3:] == 'txt':
        print('remenber to add .txt in tail')
    with open(flie_name, 'w', encoding = 'UTF-8')  as f:     # 也可使用指定路徑等方式，如： C:\A.txt
        for s in list_:
            f.write(s+'\n')                 
  
#=========================================================
#collect_imageID
#=========================================================
#there is a case i can't figure out why: when call collect_imageID(path),it show duplicate id,it seen that id_list is global variable
def collect_imageID(path,id_list=[]):    
    #id_list: photo's id
    
    photos = [f for f in os.listdir(path) if not os.path.isdir(path+'/'+f)]      
    #print(photos)      
    folders = [f for f in os.listdir(path) if os.path.isdir(path+'/'+f)]        
    #print(folders)
    #add photo id to id_list
    if photos:
        for photo in photos:
            photo_id = str(photo).split('=')[0]
            if photo_id not in id_list:                
                id_list.append(photo_id)
            else:
                print("duplicate id:{}\n".format(photo_id))
    if folders:        
        for f in folders:
            id_list = collect_imageID(path+"/"+str(f),id_list)
    
    return id_list

#=========================================================
#collect file name recurssively.
#=========================================================
def collect_fileName(path,id_list=[]):
    #id_list: photo's id
    photos = [f for f in os.listdir(path) if not os.path.isdir(path+'/'+f)]  
    #print(photos)      
    folders = [f for f in os.listdir(path) if os.path.isdir(path+'/'+f)]        
    #print(folders)
    #add photo id to id_list
    if photos:
        for photo in photos:            
                id_list.append(photo)
    if folders:
        for f in folders:
            id_list = collect_fileName(path+"/"+str(f),id_list)
    return id_list

#=========================================================
#collect file name recurssively.and store in dictionary
#=========================================================
def collect_fileName_dict(path,id_list={}):
    #id_list: photo's id
    photos = [f for f in os.listdir(path) if not os.path.isdir(path+'/'+f)]  
    #print(photos)      
    folders = [f for f in os.listdir(path) if os.path.isdir(path+'/'+f)]        
    #print(folders)
    #add photo id to id_list
    if photos:
        for photo in photos:            
                id_list[photo.split('=')[0]] = photo #use photo id to be key
    if folders:
        for f in folders:
            id_list = collect_fileName_dict(path+"/"+str(f),id_list)
    return id_list


#=========================================================
#rename if imageID not in fore 
#=========================================================
def correct_name(path):
    photos = [f for f in os.listdir(path) if not os.path.isdir(path+'/'+f)]  
    folders = [f for f in os.listdir(path) if os.path.isdir(path+'/'+f)]
    if photos:
        for photo in photos:            
            photo_id = str(photo).split('=')[0]            
            if not photo_id.isdigit():                
                pic_title,pic_id,pic_lat,pic_lon,pic_num = str(photo).split('=')[:]
                new_name = pic_id +'='+pic_title+'='+pic_lat+'='+pic_lon
                os.rename(path+'/'+str(photo),path+'/'+new_name)   
    if folders:
        for f in folders:
            correct_name(path+"/"+str(f))

def search_error_name(path):
    photos = [f for f in os.listdir(path) if not os.path.isdir(path+'/'+f)]  
    folders = [f for f in os.listdir(path) if os.path.isdir(path+'/'+f)]
    if photos:
        for photo in photos:            
            photo_id = str(photo).split('=')[0]            
            if not photo_id.isdigit():                
                print(photo)
    if folders:
        for f in folders:
            search_error_name(path+"/"+str(f))            
            
def remove_duplicate(path,id_list = [],target_path = '/home/user/python_download_image/duplicate'):
    photos = [f for f in os.listdir(path) if not os.path.isdir(path+'/'+f)]  
    folders = [f for f in os.listdir(path) if os.path.isdir(path+'/'+f)]
    if photos:
        for photo in photos:            
            photo_id = str(photo).split('=')[0]    
            if photo_id not in id_list:
                id_list.append(photo_id)        
            else:
                while(os.path.exists(target_path +'/'+photo)):
                    os.rename(path+'/'+str(photo),path+'/'+photo+'='+'0') #rename if duplicate                        
                    photo = photo+'='+'0'
                shutil.move(path+'/'+photo,target_path) #remove file to target_path
    if folders:
        for f in folders:
            remove_duplicate(path+"/"+str(f),id_list)
            
def add_label(path):
    photos = [f for f in os.listdir(path) if not os.path.isdir(path+'/'+f)]  
    folders = [f for f in os.listdir(path) if os.path.isdir(path+'/'+f)]
    if photos:
        for photo in photos:            
            pic_id,pic_title,pic_lat,pic_lon = str(photo).split('=')[:]    
            label = path.split('/')[-1]
            new_name = pic_id +'='+pic_title+'='+label+'='+pic_lat+'='+pic_lon            
            os.rename(path+'/'+str(photo),path+'/'+new_name)              
    if folders:
        for f in folders:
            add_label(path+"/"+str(f))   

#=======================================================
class BBOX:
    def  __init__(self,Id ='',label='',xmin ='-1',ymin ='-1',xmax ='-1',ymax ='-1'):      
        self.id =  Id
        self.label = label        
        self.xmin = int(float(xmin))
        self.ymin = int(float(ymin))
        self.xmax = int(float(xmax))
        self.ymax = int(float(ymax))
        
    def  __repr__( self ):
        return  'Image Object, photo id : %s , label : %s, xmin : %f , ymin : %f, xmax : %f, ymax: %f \n'\
                     % ( self.id, self .label,self.xmin,self.ymin,self.xmax,self.ymax)

#========== argument ==========================
user_name = 'user'
from xml.etree import ElementTree as et
path4info = '/home/user/python_download_image/VOCdevkit/VOC2012/Annotations'
path4image = '/home/user/python_download_image/VOCdevkit/VOC2012/JPEGImages'
file_dict = {}
collect_fileName_dict(path4info,file_dict)
file_list = list(file_dict.keys())
len(file_list)
#===============================================

data = {}
for photo_name in file_list:
    tree = et.parse(path4info+'/'+photo_name)    
    #tree = et.parse(f)     #打开xml文档 
    root = tree.getroot()         #获得root节点  
    print( "*"*10)
    filename = root.find('filename').text
    filename = filename[:-4]
    print (filename )
    #file_object = open(filename + ".txt", 'w') #写文件
    #file_object_log = open(filename + ".log", 'w') #写文件
    flag = False    
    bbox_list =[]
    num_bbox = len(root.findall('object'))
       ########################################
    for size in root.findall('size'): #找到root节点下的size节点 
        width = size.find('width').text   #子节点下节点width的值 
        height = size.find('height').text   #子节点下节点height的值 
        print (width, height)
       ########################################
       #num_bbox
        
    for object_ in root.findall('object'): #找到root节点下的所有object节点  
        name = object_.find('name').text   #子节点下节点name的值 
        print ("name",name)
        bndbox = object_.find('bndbox')      #子节点下属性bndbox的值 
        xmin = bndbox.find('xmin').text
        ymin = bndbox.find('ymin').text
        xmax = bndbox.find('xmax').text
        ymax = bndbox.find('ymax').text
      #  print( xmin, ymin, xmax, ymax)
        if name == ("bicycle" or "motorbike"):
        #file_object.write("Cyclist" + " 0 0 0 " + xmin + ".00 " + ymin + ".00 " + xmax + ".00 " + ymax + ".00 " + "0 0 0 0 0 0 0" + "\n")
 #           file_object_log.write(str(float(int(xmax) - int(xmin)) * 1920.0 / float(width)) + " " + str(float(int(ymax) - int(ymin)) * 1080.0 / float(height)) + "\n")
            flag = True
        if name == ("car"):
            #file_object.write("Car" + " 0 0 0 " + xmin + ".00 " + ymin + ".00 " + xmax + ".00 " + ymax + ".00 " + "0 0 0 0 0 0 0" + "\n")
            #file_object_log.write(str(float(int(xmax) - int(xmin)) * 1920.0 / float(width)) + " " + str(float(int(ymax) - int(ymin)) * 1080.0 / float(height)) + "\n")
            flag = True
        if name == ("person"):
            #file_object.write("Pedestrian" + " 0 0 0 " + xmin + ".00 " + ymin + ".00 " + xmax + ".00 " + ymax + ".00 " + "0 0 0 0 0 0 0" + "\n")
            #file_object_log.write(str(float(int(xmax) - int(xmin)) * 1920.0 / float(width)) + " " + str(float(int(ymax) - int(ymin)) * 1080.0 / float(height)) + "\n")
            flag = True
            #file_object.close( )   file_object_log.close()
        if name == ("dining table"):
            #file_object.write("Pedestrian" + " 0 0 0 " + xmin + ".00 " + ymin + ".00 " + xmax + ".00 " + ymax + ".00 " + "0 0 0 0 0 0 0" + "\n")
            #file_object_log.write(str(float(int(xmax) - int(xmin)) * 1920.0 / float(width)) + " " + str(float(int(ymax) - int(ymin)) * 1080.0 / float(height)) + "\n")
            flag = True
            #file_object.close( )   file_object_log.close()
        if name == ("chair"):
            #file_object.write("Pedestrian" + " 0 0 0 " + xmin + ".00 " + ymin + ".00 " + xmax + ".00 " + ymax + ".00 " + "0 0 0 0 0 0 0" + "\n")
            #file_object_log.write(str(float(int(xmax) - int(xmin)) * 1920.0 / float(width)) + " " + str(float(int(ymax) - int(ymin)) * 1080.0 / float(height)) + "\n")
            flag = True
            #file_object.close( )   file_object_log.close()
        if name == ("bottle"):
            #file_object.write("Pedestrian" + " 0 0 0 " + xmin + ".00 " + ymin + ".00 " + xmax + ".00 " + ymax + ".00 " + "0 0 0 0 0 0 0" + "\n")
            #file_object_log.write(str(float(int(xmax) - int(xmin)) * 1920.0 / float(width)) + " " + str(float(int(ymax) - int(ymin)) * 1080.0 / float(height)) + "\n")
            flag = True
            #file_object.close( )   file_object_log.close()
        if name == ("dog"):
            #file_object.write("Pedestrian" + " 0 0 0 " + xmin + ".00 " + ymin + ".00 " + xmax + ".00 " + ymax + ".00 " + "0 0 0 0 0 0 0" + "\n")
            #file_object_log.write(str(float(int(xmax) - int(xmin)) * 1920.0 / float(width)) + " " + str(float(int(ymax) - int(ymin)) * 1080.0 / float(height)) + "\n")
            flag = True
            #file_object.close( )   file_object_log.close()
        if flag:   #add bbox to list if there has object             
            label = name
            bbox_list.append(BBOX(filename,label ,xmin,ymin,xmax,ymax))
    if flag == False:  #如果没有符合条件的信息，则删掉相应的txt文件以及jpg文件
        os.remove(path4info +'/' + filename + ".xml")
        os.remove(path4image +'/'+filename +  ".jpg")
        
#    os.rename(path4image +filename +  ".jpg",new_name) 
    #os.chdir(path)# change current path
    #if not flie_name[-3:] == 'txt':
    #    print('remenber to add .txt in tail')    
    if bbox_list:
        data[filename] = bbox_list #construct mapping between name and bbox_list
        
print('step4!!!(write):  make .txt to store photo information(didn\'t include title)\n')
file_name4dataset = 'voc_dataset.txt'
#store_path = '/home/user/SY/classfy_photos/'
#os.chdir(path)# change current path
path ='/home/'+user_name+'/SY'    
if not file_name4dataset[-3:] == 'txt':
    print('remenber to add .txt in tail')
with open(path + '/' + file_name4dataset, 'w', encoding = 'UTF-8')  as f:     # 也可使用指定路徑等方式，如： C:\A.txt
    f.write("id title label lat lon xmin ymin xmax ymax\n")
    f.write("{}\n".format(str(len(data))))
    for key_ in data:
        photo = data[key_]
        f.write("{}\n".format(len(photo)))# num_bbox for one image
        for box_ in photo:
            f.write("{}={}={}={}={}={}={}={}={}\n".format(box_.id ,'title',box_.label,\
                    0.0,0.0,box_.xmin,box_.ymin,box_.xmax,box_.ymax)     )
    