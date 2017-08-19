#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 22:57:41 2017

@author: pan
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

def read_dataset(path,file_name):
    print('\n this function assume dataset are all store bbox! \n')
    complete_flie_name = os.path.join(path,file_name)
    if not os.path.exists(complete_flie_name):
        print(complete_flie_name,':no sush file name(need include path)!')
        return False
    dataset = {}
    with open(complete_flie_name, 'r',encoding = 'UTF-8') as f: 
         data = f.readlines()
    dataset_format = data[0][:-1]
    num_img = data[1][:-1]
    i = 2
    while(i<len(data)):        
        for j in range(1,int(data[i])+1):            
            info = data[i+j].split('=')            
            info[-1] = info[-1][:-1] #ignore '\n'
            dataset[info[0]] = info
            #dataset[-1][-1] = dataset[-1][-1][:-1] 
        #print("dataset[{}]:{}".format(i,data[i]))
        i+= int(data[i]) + 1        
    return dataset ,num_img,dataset_format

   #== for test above function =========
'''
user_name = 'pan' #for difference computer
region = '111'
[dataset,num_img,form] = read_dataset(path = '/home/'+user_name+'/SY/',file_name = region+'_dataset.txt')

class_set = set()
for img_id in dataset:
    dataset[img_id][2] # 2 is label
    class_set.add(dataset[img_id][2])
    
num_class = len(class_set)    
'''
import time
#=========================================================
#open .txt and read each line, then store in list
#=========================================================
def open_list_file(flie_name):
    if not os.path.exists(flie_name):
        print('no sush file name(need include path)!')
        return
    if not flie_name[-3:] == 'txt':
        print('remenber to add .txt in tail')
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
def save_list_file(list_,flie_name = 'list_'+str(int(time.time()))+'.txt',path = '/home/user/'):    
    os.chdir(path)# change current path
    if not flie_name[-3:] == 'txt':
        print('remenber to add .txt in tail')
    with open(flie_name, 'w', encoding = 'UTF-8')  as f:     # 也可使用指定路徑等方式，如： C:\A.txt
        for s in list_:
            f.write(s+'\n')                 
  
#=========================================================
#collect_imageID
#=========================================================
#there is a case i can't figure out why: when call collect_imageID(path),
#it show duplicate id,it seen that id_list is global variable
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

