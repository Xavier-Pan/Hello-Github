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

#remove_duplicate(path = '/home/user/0000GPS_photo0000' ,target_path = '/home/user/python_download_image/duplicate')
#add_label('/home/user/0000GPS_photo0000')            
#correct_name('/home/user/python_download_image/0000GPS_photo0000/0000待整理的標籤0000/0000')                
#initial_path = '/home/user/python_download_image/0000GPS_photo0000'
#initial_path = '/home/user/all_label'
#search_error_name(initial_path)
#id_list = collect_imageID(initial_path)
#manipulate_file.save_list_file(id_list,flie_name = 'download_id_with_gps(ver2).txt',path = os.getcwd())
#manipulate_file.save_list_file(id_list,flie_name = 'download_image_withGPS.txt',path = os.getcwd())
#manipulate_file.save_list_file(id_list,flie_name = 'download_imageID_noGPS_list.txt',path = os.getcwd())

#=======================================================
#delete image if it didn't appear on id_list
#=======================================================
'''
path='/home/user/python_download_image/0000GPS_photo0000/all label/艋舺龍山寺'
len(os.listdir(path))
id_list =[]
id_list = collect_imageID(path,[])
len(id_list)
save_list_file(id_list,flie_name = 'long_sang_imageID_list.txt',path = os.getcwd())
os.chdir('/home/user/python_download_image/0000GPS_photo0000/龍山寺-')
photos_2900 = os.listdir(os.getcwd())
import send2trash
for photo in photos_2900:
    photo_id = photo.split('=')[0] 
    if photo_id not in id_list:
        os.unlink('/home/user/python_download_image/0000GPS_photo0000/龍山寺-/' + photo)
       #send2trash.send2trash('/home/user/python_download_image/0000GPS_photo0000/龍山寺-/' + photo) 
sorted(id_list)


P = '/home/user/python_download_image/0000GPS_photo0000'
a = os.listdir(P)[0]
b = os.path.isdir(P+'/'+a)
photos = [f for f in os.listdir(P) if not os.path.isdir(f)]  
'''
#=======================================================
#classify photo by GPS
#=======================================================
'''
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
#=======================================================
# step1: clear class -
#=======================================================
#!!! must choose !!!
make_training_data = False#True#
region = '112' #choose which region you want to make training data
#user_name = 'user'
user_name = 'pan'
path4bbox = '/home/'+user_name+'/SY/flickr_project/BBox-Label-Tool-master/Labels/'+region#path for bbox information
path4image = '/home/'+user_name+'/SY/flickr_project/BBox-Label-Tool-master/Images/'+region#path for bbox information
   
#===========================
import json 
class BBOX( object ):
    def  __init__(self,Id ='',label='',xmin ='-1',ymin ='-1',xmax ='-1',ymax ='-1'):      
        self.id =  Id
        self.label = label        
        self.xmin = int(xmin)
        self.ymin = int(ymin)
        self.xmax = int(xmax)
        self.ymax = int(ymax)    
        
    def  __repr__( self ):
        return  'Image Object, photo id : %s , label : %s, xmin : %f , ymin : %f, xmax : %f, ymax: %f \n'\
                     % ( self.id, self .label,self.xmin,self.ymin,self.xmax,self.ymax)
                     
class Image( object ):
    def  __init__(self,Id ='',title='',label='',lat = '0',lon = '0',box_ = []):
        self.id =  Id        
        self.title =  title        
        self.label =  label
        self.lat = float(lat)
        self.lon = float(lon)
        self.bbox_list = box_        
        
    def  __repr__( self ):
        return  'Image Object ,photo id : %s , title : %s ,label : %s \n'% ( self.id, self .title,self.label)#, self.xmin,self.ymin,self.xmax,self.ymax)

if make_training_data:   
    print("make_training_data...\n")
    #=============================================
    # assume photo in ' /home/pan/SY/classfy_photos/' has all indormation(not only have ID)
    # assume photo in path4image file name didn't include title & the tile is .jpg
    #
    # step2: read photo and store in Image class 
    #        I temperary use 2 path, one for photos,one for it's bounding box information
    #        I will merge 2 path in the future,so I don't deal with the problem of photo-number-not-equal
    #=============================================
    print('step2: read photo and store in Image class \n')
    #file_list = collect_fileName(path= '/home/'+user_name+'/SY/classfy_photos/'+region,id_list=[])#for  id_,title_,label_,lat_,lon_
    file_list = collect_fileName(path= path4image,id_list=[])#for  id_,title_,label_,lat_,lon_
    file_list2 = collect_fileName(path = path4bbox,id_list=[])#has bbox information
      
    #====== make image_id ,it keep photo_id that has bbox =======
    image_id = collect_imageID(path4bbox,[])
    image_fileNames = collect_fileName(path4bbox,[])
    for fileName in image_fileNames:
        photo_id = fileName.split('=')[0]
        num_bbox = open_list_file(path4bbox +'/'+fileName)[0]
        if num_bbox == 0:#delete photo_id which no bbox
            image_id.pop(image_id.index(photo_id)) # fileName.split('=')[0] mean image-id
    #==== make Image[] object list. key is photo's id(assume photo has no title)
    data = {} 
    for i in range(len(file_list)):
        #id_,title_,label_,lat_,lon_ = file_list[i].split('=')    
        id_,label_,lat_,lon_ = file_list[i].split('=')    #assume file_name didn't include photo's title
        if id_ in image_id: #if the id are also appear in folder: BBox-Label-Tool-master/Labels
            photo = Image( Id = id_,title = "" ,label = label_, lat = lat_ ,lon = lon_[:-4])#because lon_has the form [xx.xxxxx.jpg]
            data[photo.id] = photo    
    
    #=============================================
    # step3: read photo id,label xmin, ymin, xmax, ymax from BBox-Label-Tool-master/Labels/area/
    #        store it to data{}
    #=============================================
    print('step3: read photo id,label xmin, ymin, xmax, ymax from BBox-Label-Tool-master/Labels/area/store it to data{}\n')
    print("len(data.keys())={}\n".format(len(data.keys())))
    
    #bbox_id = []
    for file_name in file_list2:
        photo_id =  file_name.split('=')[0]     
        if photo_id in data.keys():   #to avoid create a new data[photo_id] which has no information of id,label,lat,lon
            complete_name = path4bbox +'/'+file_name                        
            lines = open_list_file(complete_name)
            num_bbox = int(lines[0])
            box_list = []
            for index in range(1,num_bbox+1):
                label, xmin, ymin, xmax, ymax = lines[index].split(' ')#read bbox information        
                box_list.append(BBOX(photo_id,label ,xmin,ymin,xmax,ymax))
            if box_list:
                data[photo_id].bbox_list = box_list
            #add bbox information to image !!!you ccan't just use append(). it will cause share bbox_list    
    
    #=============================================
    # !!!(write)step4:  make .txt to store photo information
    #
    #=============================================
    print('step4!!!(write):  make .txt to store photo information(didn\'t include title)\n')
    file_name = region + '_dataset.txt'
    #store_path = '/home/user/SY/classfy_photos/'
    #os.chdir(path)# change current path
    path ='/home/'+user_name+'/SY'    
    if not file_name[-3:] == 'txt':
        print('remenber to add .txt in tail')
    with open(path + '/' + file_name, 'w', encoding = 'UTF-8')  as f:     # 也可使用指定路徑等方式，如： C:\A.txt
        f.write("id title label lat lon xmin ymin xmax ymax\n")
        f.write("{}\n".format(str(len(data))))
        for key_ in data:
            photo = data[key_]
            f.write("{}\n".format(len(photo.bbox_list)))# num_bbox for one image
            for box_ in photo.bbox_list:
                f.write("{}={}={}={}={}={}={}={}={}\n".format(photo.id,'title',box_.label,\
                        photo.lat,photo.lon,box_.xmin,box_.ymin,box_.xmax,box_.ymax)     )
    
    #=========================================================
    # step5 change file name (only keep id) because the SSD code assume file names are equal to it's name in dictionary
    #=========================================================
    print('step5: change file name (only keep id) because the SSD code assume file names are equal to it\'s name in dictionary\n')
    path=path4image  
    #id_list: photo's id
    photos_name = [f for f in os.listdir(path) if not os.path.isdir(path+'/'+f)]  
    #print(photos)      
    #add photo id to id_list
    if photos_name:
        for name in photos_name:        
            new_name = path + '/' + name.split('=')[0]#if name is already only id,then newname didn't change
            os.rename(path + '/'+ name,new_name)   

#============================================
# step6: make training data 
# 
#=============================================
print('step6: make training data \n')
#============== read file ===============  
import scipy
#path ='/home/'+user_name+'/SY/classfy_photos/'+region
path = path4image
#photos = [f for f in os.listdir(path ) if not os.path.isdir(path+'/'+f)]   
photos = {}
for f_name in os.listdir(path ):
    if not os.path.isdir(path+'/'+f_name):
        photos[f_name.split('=')[0]] = f_name

#====== count number of class =========
class_set = {''}
lines = open_list_file('/home/'+user_name+'/SY/'+region + '_dataset.txt')
i = 2
while i < len(lines):
    box_num = int(lines[i])
    if box_num > 0:
        for j in range(1,box_num+1):
            class_set.add(lines[i+j].split('=')[2])#add label
    i += box_num+1
        
num_class = len(class_set)-1 #minus {''}
class_set.remove('')
#============== make training data ===============  !!!note:class_set has no order and label2prob's order is by dictionary
# make one hot bector
label2prob = {}
prob = np.eye(num_class)
for i,class_name in enumerate(class_set):
    label2prob[class_name]=list(prob[i])
    #'cksHell':[1.,0.,0.,0.],'concertHell':[0.,1.,0.,0.],'theaterHell':[0.,0.,1.,0.],'taipei101':[0.,0.,0.,1.]}
    print('\n')
    print(label2prob)
    print('\n\n')
train_data = {}
file_name = '/home/'+user_name+'/SY/'+ region +'_dataset.txt'
if not file_name[-3:] == 'txt':
    print('remenber to add .txt in tail')
with open(file_name, 'r', encoding = 'UTF-8')  as f:     # 也可使用指定路徑等方式，如： C:\A.txt
    f.readline()#the discript of format
    size = int(f.readline())
    for i in range(size):
        num_box = int(f.readline())
        if num_box:
            boxes = np.zeros([num_box,4+num_class])
            for j in range(num_box):
                
                [id_, title, label, lat, lon, xmin, ymin, xmax, ymax] = f.readline().split('=')        
                #img = np.shape(scipy.misc.imread(path+'/'+photos[0].split('=')[0]))
                height,weight,channel = np.shape(scipy.misc.imread(path+'/'+id_))#get image shape
                
                boxes[j] = np.array([float(xmin)/weight,float(ymin)/height,float(xmax)/weight,float(ymax)/height]+label2prob[label])
                                 #[xmin,ymin,xmax,ymax,prob(y1),prob(y2),prob(y3)]
            train_data[id_] = boxes
                  
#!!! note !!!
#if you use np.shape(np.array([i for i in range(8)])) ,then you get (8,)
#if you use np.shape(np.zeros( [1,8] )) , then you get (1,8)
#=========================================================
#change file name (add .jpg) when use BBOX-label application
#=========================================================
'''
path='/home/'+user_name+'/SY/flickr_project/BBox-Label-Tool-master/Images/' + region
new_path = '/home/'+user_name+'/SY/flickr_project/BBox-Label-Tool-master/Images'
photos_name = [f for f in os.listdir(path) if not os.path.isdir(path+'/'+f)]

#=== to remove the photo which didn't appear in label
#flie_name ='/home/'+user_name+'/Desktop/label33'
#need_delete_list_ = open_list_file(flie_name)
#for i in range(len(need_delete_list_)):
#    need_delete_list_[i] = need_delete_list_[i].split(' ')[-1]


#for s in photos_name:
#    if s.split('=')[0] in need_delete_list_:
#        shutil.move(path+'/'+s,new_path)#remove
        #os.unlink(path + '/' + s)# delete
'''
#===== change name to add .jpg ====

def fileNameAddJpg(photos_name,path):
    if photos_name:
        for name in photos_name:        
            new_name = path + '/' + name + '.jpg'
            os.rename(path + '/'+ name,new_name) 

#fileNameAddJpg(photos_name,path)

'''
#===== take off .jpg ====
#path='/home/'+user_name+'/SY/flickr_project/BBox-Label-Tool-master/Images/'+region
#photos_name = [f for f in os.listdir(path) if not os.path.isdir(path+'/'+f)]
#if photos_name:
#    for name in photos_name:        
#        new_name = path + '/' + name.split('.')[0]
#        os.rename(path + '/'+ name,new_name) 
'''

#=========================================================
# make .txt to store all photo for search  (dataset.txt location:= '/home/user/python_download_image/0000GPS_photo0000')
#=========================================================
'''
path4photo = '/home/'+user_name+'/python_download_image/0000GPS_photo0000/all label'
#collect all file name include in path( recursive)
id_list = collect_fileName_dict(path4photo, {})
len(id_list)
#len(all_data_list)
new_dataset = []
path4txt = '/home/'+user_name+'/python_download_image/0000GPS_photo0000'
all_data_list = open_list_file(flie_name = path4txt+'/all_photo_file_name.txt')
for data_ in all_data_list:
    [id_,title_,lat_,lon_ ]= data_.split('=')
    if id_ in id_list.keys():
        label_ = id_list[id_].split('=')[1]
        new_dataset.append(id_+'='+ title_ +'='+ label_ +'='+ lat_ +'='+ lon_)
    else:
        print("id miss:{}\n".format(id_))
    
save_list_file(new_dataset,flie_name = 'all_photo_dataset.txt',path = path4txt)
'''
#=============================
#delete title from file name
'''
def changeName(path):
    photos = [f for f in os.listdir(path) if not os.path.isdir(path+'/'+f)]      
    #print(photos)      
    folders = [f for f in os.listdir(path) if os.path.isdir(path+'/'+f)]        

    if photos:
        for photo in photos:
            id_,title_,lat_,lon_ = photo.split('=')
            label_ = path.split('/')[-1]
            file_name_ = id_+'='+label_+'='+lat_+'='+lon_
            os.rename(path + '/' + photo , path + '/' + file_name_)
            
    if folders:        
        for f in folders:            
            changeName(path+'/'+f)

changeName(path)
'''
#====== count how many image has no bbox in training data =====
'''
path = '/home/pan/SY/112_dataset.txt'
test_list = open_list_file(path)
count_zero_bbox = 0;
for bbox_t in test_list:
    if bbox_t[0] =='0':
        count_zero_bbox+=1
        print(bbox_t)
print("totally has {} 's zero".format(count_zero_bbox))
'''