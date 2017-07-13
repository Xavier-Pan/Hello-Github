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
import manipulate_file 
max_lat = 26.
max_lon = 122.
lat = [0,25.024439,25.034215,25.042341,25.052082,25.061537,25.070958,max_lat]
lon = [0,121.495034,121.510641,121.527292,121.543331,121.559898,121.575844,max_lon]
lat = np.asarray(lat)
lon = np.asarray(lon)
homePath = os.getcwd()

#collect_imageID
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
                print(photo_id)
    if folders:
        for f in folders:
            id_list = collect_imageID(path+"/"+str(f),id_list)
    return id_list


#rename if imageID not in fore 
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


#remove_duplicate(path = '/home/user/0000GPS_photo0000' ,target_path = '/home/user/python_download_image/duplicate')
#add_label('/home/user/0000GPS_photo0000')            
#correct_name('/home/user/python_download_image/0000GPS_photo0000/0000待整理的標籤0000/0000')                
#initial_path = '/home/user/python_download_image/0000GPS_photo0000'
initial_path = '/home/user/all_label'
#search_error_name(initial_path)
id_list = collect_imageID(initial_path)
manipulate_file.save_list_file(id_list,flie_name = 'download_id_with_gps(ver2).txt',path = os.getcwd())
#manipulate_file.save_list_file(id_list,flie_name = 'download_image_withGPS.txt',path = os.getcwd())
#manipulate_file.save_list_file(id_list,flie_name = 'download_imageID_noGPS_list.txt',path = os.getcwd())
'''
P = '/home/user/python_download_image/0000GPS_photo0000'
a = os.listdir(P)[0]
b = os.path.isdir(P+'/'+a)
photos = [f for f in os.listdir(P) if not os.path.isdir(f)]  
'''
#classify photo by GPS
photos_folder = initial_path#'/home/user/0000GPS_photo0000'
folders = [f for f in os.listdir(photos_folder)] 
m = np.zeros([8,8])
for folder in folders:
    print(folder)
    label = folder
    photos = [f for f in os.listdir(os.path.join(photos_folder,folder)) if not os.path.isdir(photos_folder+'/'+folder+'/'+f)]        
    #print(photos)
    for photo in photos:
      #  print(photo)
        lat_,lon_ = str(photo).split('=')[-2:]
         #lon_ = str(photo).split('=')[-1]
        first_name ,last_name= 0,0
        #classify director,
        for i in range(len(lat)):
            if float(lat_) < float(lat[i]):
                first_name = i
                break
        for i in range(len(lon)):
            if float(lon_) < float(lon[i]):
                last_name = i
                break        
        new_folder = str(first_name)+str(last_name)
        m[first_name][last_name]+=1

        if not os.path.exists(str(new_folder)):
            os.mkdir(str(new_folder))
        folder_path = os.path.join(photos_folder,str(folder))        
        photo_path = os.path.join(folder_path,str(photo))
        shutil.copy(photo_path,new_folder)
      #  print("photo_path:",photo_path)
        pic_id,pic_title,pic_lat,pic_lon = str(photo).split('=')[:]
        photo_name = pic_id+'='+pic_title+'='+label+'='+pic_lat+'='+pic_lon
        new_name = new_folder+"/"+photo_name
        os.rename(new_folder+'/'+str(photo),new_name)   
        #dir_list = os.listdir(folder_path)
        #print("dir_list:",dir_list)
    for i in range(8):
        for j in range(8):
            if m[i][j] > 50:
                print('[{0}][{1}]={2}'.format(i,j,m[i][j]))
    m = np.zeros([8,8])

    



'''
files = [f for f in os.listdir('./0000GPS_photo0000') if os.path.isfile(f)]

for f in files:
        print(f)

        
files = []
for f in os.listdir('./final_project/photo'):
#        if os.path.isfile(f):
    files.append(f)
for f in files:
        print( f)
'''        