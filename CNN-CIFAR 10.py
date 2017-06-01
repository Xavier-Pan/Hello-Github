# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 08:27:39 2017

@author: User
"""

import time
tStart = time.time()
import data_helpers
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Parameters
learning_rate = 1
training_epochs = 80

batch_size = 100
display_step = 10
initial_w=0
keep_prob_=1;
# Network Parameters
n_hidden_1 = 700 # 1st layer number of features
n_hidden_2 = 400 # 2nd layer number of features
n_hidden_3 = 400 # 3nd layer number of features
n_input = 32*32*3 # CIFAR 10 data input (img shape: 32*32)
n_classes = 10 # MNIST total classes (0-9 digits)
img_size=32
num_channels=3
#===============preprocess data================================================

data_sets = data_helpers.load_data()
num_train_data=int(np.shape(data_sets['labels_train'])[0])
num_test_data=int(np.shape(data_sets['labels_test'])[0])
# make one-hot vector
train_label=np.zeros([num_train_data,n_classes])
test_label=np.zeros([num_test_data,n_classes])
for i in range(num_train_data):        
    train_label[i,data_sets['labels_train'][i]]=1
for i in range(num_test_data):    
    test_label[i,data_sets['labels_test'][i]]=1

data_sets['labels_train']=train_label
data_sets['labels_test']=test_label
tB=np.zeros([50000,32,32,3])
tB2=np.zeros([10000,32,32,3])
#transfer [batch,channel,H,W] to [batch,H,W,channel]
X_test=np.reshape(data_sets['images_test'], (-1, 3, 32, 32))
X_train=np.reshape(data_sets['images_train'], (-1, 3, 32, 32))
np.copyto(tB[:,:img_size,:img_size,0],X_train[:,0,:img_size,:img_size])
np.copyto(tB[:,:img_size,:img_size,1],X_train[:,1,:img_size,:img_size])
np.copyto(tB[:,:img_size,:img_size,2],X_train[:,2,:img_size,:img_size])
data_sets['images_train']=np.reshape(tB,(50000,-1))

np.copyto(tB2[:,:img_size,:img_size,0],X_test[:,0,:img_size,:img_size])
np.copyto(tB2[:,:img_size,:img_size,1],X_test[:,1,:img_size,:img_size])
np.copyto(tB2[:,:img_size,:img_size,2],X_test[:,2,:img_size,:img_size])
data_sets['images_test']=np.reshape(tB2,(10000,-1))

#==============================================================================



# tf Graph input
x = tf.placeholder("float", shape=[None, img_size*img_size*num_channels])
y = tf.placeholder("float", shape=[None,n_classes])
h1_result = tf.Variable(np.zeros([ 32,32,32]))
h2_result = tf.Variable(np.zeros([ 32,32,32]))

keep_prob = tf.placeholder(tf.float32) #probability of ckeep the neurons
hid_idx={'h1','h2','h3','h4','h5'}
# Create model

def convolutional_neural_network(x,weights,biases,keep_prob):
    x = tf.reshape(x, [-1,32,32,3])
    conv1 = tf.add(tf.nn.conv2d(x, weights['w_conv1'], strides=[1,1,1,1], padding='SAME'), biases['b_conv1'])
    conv1 = tf.nn.relu(conv1)
#    conv1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')  
#    conv1 = tf.nn.dropout(conv1,keep_prob=0.9)
    
    conv2 = tf.add(tf.nn.conv2d(conv1, weights['w_conv2'], strides=[1,1,1,1], padding='SAME'), biases['b_conv2'])
    conv2 = tf.nn.relu(conv2)    
    conv2 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

#    conv3 = tf.add(tf.nn.conv2d(conv2, weights['w_conv3'], strides=[1,1,1,1], padding='SAME'), biases['b_conv3'])
#    conv3 = tf.nn.relu(conv3)
#    conv3 = tf.nn.max_pool(conv3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')  
   
    fc = tf.reshape(conv2, [-1,16*16*32])    
    fc = tf.nn.relu(tf.add(tf.matmul(fc, weights['w_fc']), biases['b_fc'])) 
    fc2 = tf.nn.relu(tf.add(tf.matmul(fc, weights['w_fc2']), biases['b_fc2']))    
    fc2 = tf.nn.dropout(fc2,keep_prob=0.9)
    out_layer = tf.add(tf.matmul(fc2, weights['out']), biases['out'])
    return out_layer
  
def get_featuremap(x,weights,biases,keep_prob):
    x = tf.reshape(x, [-1,32,32,3])
    conv1 = tf.add(tf.nn.conv2d(x, weights['w_conv1'], strides=[1,1,1,1], padding='SAME'), biases['b_conv1'])
    conv1 = tf.nn.relu(conv1)
#    feature_map['h1']=conv1
 #   conv1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')  
    #conv1 = tf.nn.dropout(conv1,keep_prob)
    conv2 = tf.add(tf.nn.conv2d(conv1, weights['w_conv2'], strides=[1,1,1,1], padding='SAME'), biases['b_conv2'])
    conv2 = tf.nn.relu(conv2) 
    t_conv2=conv2
 #   feature_map['h2']=conv2
    conv2 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    return conv1,t_conv2

# Store layers weight & bias
weights = {'w_conv1':tf.Variable(tf.random_normal([3,3,3,32]),name='w_conv1'),
           'w_conv2':tf.Variable(tf.random_normal([3,3,32,32]),name='w_conv2'),
      #     'w_conv3':tf.Variable(tf.random_normal([5,5,64,64]),name='w_conv3'),
           'w_fc':tf.Variable(tf.random_normal([16*16*32,1024]),name='w_fc'),
           'w_fc2':tf.Variable(tf.random_normal([1024,1024])),
           'out':tf.Variable(tf.random_normal([1024,n_classes],name='w_out'))
           }
 
biases = {'b_conv1':tf.Variable(tf.random_normal([32]),name='b_conv1'),
          'b_conv2':tf.Variable(tf.random_normal([32]),name='b_conv2'),
      #    'b_conv3':tf.Variable(tf.random_normal([64]),name='b_conv3'),
          'b_fc':tf.Variable(tf.random_normal([1024]),name='b_fc'),
          'b_fc2':tf.Variable(tf.random_normal([1024])),
          'out':tf.Variable(tf.random_normal([n_classes]),name='b_out')
          }

# Construct model
pred = convolutional_neural_network(x,weights,biases,keep_prob)
get_feature = get_featuremap(x,weights,biases,keep_prob)
# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)


# Initializing the variables
init = tf.global_variables_initializer()
record_w=np.zeros([n_input, n_hidden_1])
record_w2=np.zeros([n_hidden_1, n_hidden_2])
record_w3=np.zeros([n_hidden_2, n_hidden_3])
record_wout=np.zeros([n_hidden_2, n_hidden_3])

error_cost = np.zeros([training_epochs])
test_err=np.zeros([training_epochs+1])
train_err=np.zeros([training_epochs+1])
h1_result_=np.zeros([32,32,32])
h2_result_=np.zeros([32,32,32])
get_feature_map = {
               'h1':np.zeros([32,32,32]),
               'h2':np.zeros([32,32,32])
               }

total_data_size=data_sets['images_train'].shape[0]
indices = np.random.choice(a=total_data_size,size=total_data_size,replace=False)#data index for doing batch
#saver = tf.train.Saver() #for save variable
# Launch the graph
with tf.Session() as sess:
    sess.run(init)         
    saver = tf.train.Saver({"w_conv1": weights['w_conv1'],
                            "w_conv2": weights['w_conv2'],
                            "w_fc": weights['w_fc'],
                            "w_out": weights['out'],
                            "b_conv1": biases['b_conv1'],
                            "b_conv2": biases['b_conv2'],
                            "b_fc": biases['b_fc'],
                            "b_out": biases['out']})
   # save_path = saver.save(sess, "/tmp/model.ckpt")#save all initial variable
    save_path = saver.save(sess, "/tmp/cnn_kernel77.ckpt")#save all initial variable
  #  saver.restore(sess, "/tmp/model.ckpt")#restore all variable       
    record_conv1=weights['w_conv1'].eval()
    record_conv2=weights['w_conv2'].eval()
    print('conv1[:,:,0,0]:')    

    print(record_conv1[:,:,0,0])
    print(record_conv2[:,:,0,0])
    train_size=np.shape(data_sets['images_train'])[0]
    test_size=np.shape(data_sets['images_test'])[0]
    input_size=2000;
    #record the error before training
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    for i in range(np.int(10000/input_size)):# memory can't load all for once
            test_err[0]+=1-accuracy.eval({x: data_sets['images_test'][input_size*i:input_size*(i+1),:], y: data_sets['labels_test'][input_size*i:input_size*(i+1),:],keep_prob: 1})#!!!!!!!!!
    test_err[0]/=test_size/input_size
    
    for i in range(np.int(50000/input_size)):# memory can't load all for once
           train_err[0]+=1-accuracy.eval({x: data_sets['images_train'][input_size*i:input_size*(i+1),:], y: data_sets['labels_train'][input_size*i:input_size*(i+1)],keep_prob: 1})
    train_err[0]/=train_size/input_size  
  #  test_err[0]=1-accuracy.eval({x: data_sets['images_test'], y: data_sets['labels_test'],keep_prob: 1})
  #  for i in range(5):# memory can't load all for once
  #      train_err[0]+=1-accuracy.eval({x: data_sets['images_train'][10000*i:10000*(i+1),:], y: data_sets['labels_train'][10000*i:10000*(i+1)],keep_prob: 1})
  #  train_err[0]/=5
    
    # Training cycle

    for epoch in range(training_epochs):
        avg_cost = 0.
        total_cost=0.    
        total_batch = int(total_data_size/batch_size) #!!!!!!
        # Loop over all batches        
        for i in range(total_batch):                                
            batch_x = data_sets['images_train'][indices[(i)*batch_size:(i+1)*batch_size],:]
            
            batch_y = data_sets['labels_train'][ indices[(i)*batch_size:(i+1)*batch_size],:]
        #    print("np.shape(batch_y):",np.shape(batch_y))
                                  
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y,
                                                          keep_prob:keep_prob_})
            # Compute average loss
            avg_cost += c / total_batch
                
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        for i in range(np.int(10000/input_size)):# memory can't load all for once
            test_err[epoch+1]+=1-accuracy.eval({x: data_sets['images_test'][input_size*i:input_size*(i+1),:], y: data_sets['labels_test'][input_size*i:input_size*(i+1),:],keep_prob: 1})#!!!!!!!!!
        test_err[epoch+1]/=test_size/input_size

        for i in range(np.int(50000/input_size)):# memory can't load all for once
           train_err[epoch+1]+=1-accuracy.eval({x: data_sets['images_train'][input_size*i:input_size*(i+1),:], y: data_sets['labels_train'][input_size*i:input_size*(i+1)],keep_prob: 1})
        train_err[epoch+1]/=train_size/input_size  
        #train_err[epoch+1]=1-accuracy.eval({x: data_sets['images_train'][1:10000,:], y: data_sets['labels_train'][1:10000],keep_prob: 1})#!!!!!!!!!
        if epoch % display_step == 0:
            print('epoch:{:5d} '.format(epoch+1),"test_accurate:{:3f}".format(1-test_err[epoch+1])," train_accurate:{:5f}".format(1-train_err[epoch+1]))

        record_wout=weights['out'].eval()
    print("Optimization Finished!") 
    print(h1_result_[:,:,0])
    h1_result_,h2_result_=sess.run(get_feature,feed_dict={x: data_sets['images_test'][:10,:],
                                     y: data_sets['labels_test'][:10,:],                                     
                                     keep_prob:keep_prob_})
    t=0;
    for i in range(np.int(test_size/input_size)):# memory can't load all for once
        t+=accuracy.eval({x: data_sets['images_test'][input_size*i:input_size*(i+1),:], y: data_sets['labels_test'][input_size*i:input_size*(i+1),:],keep_prob: 1})
    print("Accuracy:",t/(test_size/input_size) )#!!!!!!!!!  

   
    
print('max accurate:',np.max(1-test_err))
#plt.xlim(0,training_epochs)
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(np.arange(1,training_epochs), train_err[1:training_epochs],label='training error')
ax.plot(np.arange(1,training_epochs), test_err[1:training_epochs],label='test error')
ax.legend()
plt.title("learning curve")
plt.xlabel("epoch")
plt.ylabel("error rate")
#plt.ylim(0,5)
plt.ion()
plt.show()   
print("================data=====================")
traing_accuracy=1-train_err[training_epochs]
print("traing accuracy:",round(traing_accuracy,3)," traing error:",round(train_err[training_epochs],4))
print("test accuracy:",round(1-test_err[training_epochs],4),'test error:',round(test_err[training_epochs],4))
print( "keep prob:",round(keep_prob_,2))
print("=========================================")
tEnd = time.time()
#print('runing time:',np.floor(round(tEnd-tStart,2)/60),'m.',round(round(tEnd-tStart,2)%60),'sec.')
print('Total time: {:5.2f}s'.format(tEnd-tStart)) 



from scipy.misc import toimage
# load data
# create a grid of 3x3 images
X_train=np.reshape(data_sets['images_test'], (-1,32, 32,3))
h=w=32
k=0
for i in range(k,k+9):
    plt.subplot(3,3,(i-k)%9+1)
    plt.imshow(toimage([X_train[i,:h,:w,0],X_train[i,:h,:w,1],X_train[i,:h,:w,2]]))#plt.imshow(toimage([tB[i,:h,:w,0],tB[i,:h,:w,1],tB[i,:h,:w,2]]))    
print(np.shape(data_sets['images_test']))

j=0
k=0
for i in range(k,k+9):
    plt.subplot(3,3,(i-k)%9+1)
    j=i
    plt.imshow(toimage([h2_result_[4,:,:,j],h2_result_[4,:,:,j],h2_result_[4,:,:,j]]))
print(np.shape(h2_result_))
for i in range(k,k+9):
    plt.subplot(3,3,(i-k)%9+1)
    j=i  
    plt.imshow(toimage([h1_result_[4,:,:,j],h1_result_[4,:,:,j],h1_result_[4,:,:,j]]))
