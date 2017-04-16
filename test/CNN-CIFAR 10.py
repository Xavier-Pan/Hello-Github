# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 08:27:39 2017

@author: User
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""
#from __future__ import print_function

# Import MNIST data
import time
tStart = time.time()
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
import data_helpers
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
# Parameters
data_sets = data_helpers.load_data()
learning_rate = .0005
training_epochs = 50
batch_size = 100
display_step = 10
initial_w=0
keep_prob_=1;
# Network Parameters
n_hidden_1 = 700 # 1st layer number of features
n_hidden_2 = 400 # 2nd layer number of features
n_hidden_3 = 400 # 3nd layer number of features
num_train_data=int(np.shape(data_sets['labels_train'])[0])
num_test_data=int(np.shape(data_sets['labels_test'])[0])
n_input = 32*32*3 # CIFAR 10 data input (img shape: 32*32)
n_classes = 10 # MNIST total classes (0-9 digits)
img_size=32
num_channels=3
lamda=0/(n_input*n_hidden_1+n_hidden_1*n_hidden_2+n_hidden_2*n_classes) #for 2 layer
#lamda=70/(n_input*n_hidden_1+n_hidden_1*n_hidden_2+n_hidden_2*n_hidden_2+n_hidden_2*n_classes) #for 3 layer
#lamda=0/(n_input*n_hidden_1+n_hidden_1*n_hidden_2+n_hidden_2*n_hidden_2+n_hidden_2*n_hidden_2+n_hidden_2*n_classes) #for 4 layer

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




# tf Graph input
x = tf.placeholder("float", shape=[None, img_size*img_size*num_channels])
y = tf.placeholder("float", shape=[None,n_classes])
keep_prob = tf.placeholder(tf.float32) #probability of ckeep the neurons
hid_idx={'h1','h2','h3','h4','h5'}
# Create model
def convolutional_neural_network(x,weights,biases,keep_prob):
    x = tf.reshape(x, [-1,32,32,3])
    conv1 = tf.add(tf.nn.conv2d(x, weights['w_conv1'], strides=[1,1,1,1], padding='SAME'), biases['b_conv1'])
    conv1 = tf.nn.relu(conv1)
    conv1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')  
    #conv1 = tf.nn.dropout(conv1,keep_prob)
    conv2 = tf.add(tf.nn.conv2d(conv1, weights['w_conv2'], strides=[1,1,1,1], padding='SAME'), biases['b_conv2'])
    conv2 = tf.nn.relu(conv2)
    conv2 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    
    
    fc = tf.reshape(conv2, [-1,8*8*64])    
    fc = tf.nn.relu(tf.add(tf.matmul(fc, weights['w_fc']), biases['b_fc']))
    fc = tf.nn.dropout(fc, keep_prob)
    # dropout剔除一些"神经元"
     #fc = tf.nn.dropout(fc, 0.8) 
    out_layer = tf.add(tf.matmul(fc, weights['out']), biases['out'])
    return out_layer 
# Store layers weight & bias
weights = {'w_conv1':tf.Variable(tf.random_normal([5,5,3,32]),name='w_conv1'),
           'w_conv2':tf.Variable(tf.random_normal([5,5,32,64]),name='w_conv2'),
           'w_fc':tf.Variable(tf.random_normal([8*8*64,1024]),name='w_fc'),
           #'w_fc2':tf.Variable(tf.random_normal([8*8*64,1024])),
           'out':tf.Variable(tf.random_normal([1024,n_classes],name='out'))
           }
 
biases = {'b_conv1':tf.Variable(tf.random_normal([32]),name='b_conv1'),
          'b_conv2':tf.Variable(tf.random_normal([64]),name='b_conv2'),
          'b_fc':tf.Variable(tf.random_normal([1024]),name='b_fc'),
         # 'b_fc2':tf.Variable(tf.random_normal([1024])),
          'out':tf.Variable(tf.random_normal([n_classes]),name='b_out')
          }

# Construct model
pred = convolutional_neural_network(x,weights,biases,keep_prob)


cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
# Define loss and optimizer
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
optimizer = tf.train.AdamOptimizer(learning_rate=0.0005).minimize(cost)


# Initializing the variables
init = tf.global_variables_initializer()
record_w=np.zeros([n_input, n_hidden_1])
record_w2=np.zeros([n_hidden_1, n_hidden_2])
record_w3=np.zeros([n_hidden_2, n_hidden_3])
record_wout=np.zeros([n_hidden_2, n_hidden_3])
# Launch the graph
error_cost = np.zeros([training_epochs])
test_err=np.zeros([training_epochs+1])
train_err=np.zeros([training_epochs+1])
'''
prepare data
'''
total_data_size=data_sets['images_train'].shape[0]
indices = np.random.choice(a=total_data_size,size=total_data_size,replace=False)#data index for doing batch
saver = tf.train.Saver() #for save variable

with tf.Session() as sess:
 #   saver.restore(sess, "/tmp/model.ckpt")#restore all variable
    sess.run(init)    
    save_path = saver.save(sess, "/tmp/model.ckpt")#save all initial variable
   
    record_conv1=weights['w_conv1'].eval()
    print('conv1[:,:,0,0]:')
    print(record_conv1[:,:,0,0])
    
    #record the error before training
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    test_err[0]=1-accuracy.eval({x: data_sets['images_test'], y: data_sets['labels_test'],keep_prob: 1})
    for i in range(5):# memory can't load all for once
        train_err[0]+=1-accuracy.eval({x: data_sets['images_train'][10000*i:10000*(i+1),:], y: data_sets['labels_train'][10000*i:10000*(i+1)],keep_prob: 1})
    train_err[0]/=5
    
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_cost=0.    
        total_batch = int(total_data_size/batch_size) #!!!!!!
        # Loop over all batches        
        for i in range(total_batch):                                
            batch_x = data_sets['images_train'][indices[(i)*batch_size:(i+1)*batch_size],:]
            batch_y = data_sets['labels_train'][ indices[(i)*batch_size:(i+1)*batch_size],:]
            np.shape(data_sets['images_train'][ indices[(i)*batch_size:(i+1)*batch_size],:])          
                                  
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y,
                                                          keep_prob:keep_prob_})
            # Compute average loss
            avg_cost += c / total_batch
                
        co = sess.run(cost, feed_dict={x:data_sets['images_test'],y: data_sets['labels_test'],keep_prob: 1})#!!!!!!
        total_cost = co 
        # Display logs per epoch step        
        error_cost[epoch]=avg_cost
    #    if epoch % display_step == 0:
     #       print("Epoch:", '%04d' % (epoch+1), "cost=","{:.9f}".format(avg_cost)," Total_cost:",total_cost) 
            # Test model
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        test_err[epoch+1]=1-accuracy.eval({x: data_sets['images_test'], y: data_sets['labels_test'],keep_prob: 1})#!!!!!!!!!
        for i in range(5):# memory can't load all for once
            train_err[epoch+1]+=1-accuracy.eval({x: data_sets['images_train'][10000*i:10000*(i+1),:], y: data_sets['labels_train'][10000*i:10000*(i+1)],keep_prob: 1})
        train_err[epoch+1]/=5
        #train_err[epoch+1]=1-accuracy.eval({x: data_sets['images_train'][1:10000,:], y: data_sets['labels_train'][1:10000],keep_prob: 1})#!!!!!!!!!
        if epoch % display_step == 0:
            print('epoch:{:5d} '.format(epoch+1),"test_accurate:",1-test_err[epoch+1]," train_accurate:",1-train_err[epoch+1])
 #       record_w=weights['h1'].eval(sess)
 #       record_w2=weights['h2'].eval(sess)
 #       record_w3=weights['h3'].eval(sess)
        record_wout=weights['out'].eval()
    print("Optimization Finished!") 
    print("Accuracy:", accuracy.eval({x: data_sets['images_test'], y: data_sets['labels_test'],keep_prob: 1}))#!!!!!!!!!  
   
    
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
print("lambda:",round(lamda,6),"keep prob:",round(keep_prob_,2))
print("=========================================")
tEnd = time.time()
#print('runing time:',np.floor(round(tEnd-tStart,2)/60),'m.',round(round(tEnd-tStart,2)%60),'sec.')
print('Total time: {:5.2f}s'.format(tEnd-tStart)) 



from scipy.misc import toimage
# load data
# create a grid of 3x3 images
X_train=np.reshape(data_sets['images_test'], (-1,32, 32,3))
h=w=32
k=78;
for i in range(k,k+9):
    plt.subplot(3,3,(i-k)%9+1)
    plt.imshow(toimage([X_train[i,:h,:w,0],X_train[i,:h,:w,1],X_train[i,:h,:w,2]]))#plt.imshow(toimage([tB[i,:h,:w,0],tB[i,:h,:w,1],tB[i,:h,:w,2]]))    

