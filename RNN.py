# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 01:00:39 2017

@author: SY-Pan
"""

import time
tStart = time.time()
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

imdb = np.load('imdb_word_emb.npz')
X_train = imdb['X_train']
y_train_ = imdb['y_train']
X_test  = imdb['X_test']
y_test_  = imdb['y_test']
test_size=np.shape(y_test_)[0]
train_size=np.shape(y_train_)[0]
y_train=np.zeros([train_size,2])
y_test=np.zeros([test_size,2])

class_num=2
identity_mat=np.identity(class_num)
#===pre process data =====================
for i in range(test_size):
    y_test[i]=identity_mat[y_test_[i],:]
    y_train[i]=identity_mat[y_train_[i],:]
#=========================================

num_hidd_unit=256
data_dim=128
series_len=80
batch_size=100
training_epochs=200
learning_rate=0.01
display_step=10
weights= {
         'W': tf.Variable(tf.random_normal([num_hidd_unit,num_hidd_unit])),
         'V': tf.Variable(tf.random_normal([num_hidd_unit,class_num])),
         'U': tf.Variable(tf.random_normal([data_dim,num_hidd_unit])),       
         }
         
biases= {
         'b': tf.Variable(tf.random_normal([1,num_hidd_unit])),
         'c': tf.Variable(tf.random_normal([1,class_num]))
}

x=tf.placeholder("float",shape=[None,series_len,data_dim])
y=tf.placeholder("float",shape=[None,class_num])
#print("OK!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

def RNN(x, weights, biases):
    h = tf.zeros(tf.shape(tf.matmul(x[:,0,:], weights['U'])))# get h's shape    
    for i in range(series_len):          
        xU = tf.matmul(x[:,i,:], weights['U'])
        hW = tf.matmul(h, weights['W'])  
        #update state
        a=tf.add(tf.add(xU,hW),biases['b'])
        h=tf.nn.tanh(a)                
    hV = tf.add(tf.matmul(h, weights['V']),biases['c'])
    return hV
    
pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)) 
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

#initial
error_cost = np.zeros([training_epochs])
test_err=np.zeros([training_epochs+1])
train_err=np.zeros([training_epochs+1])
init = tf.global_variables_initializer()
#print("Session start!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
input_size=5000
with tf.Session() as sess:
    sess.run(init)    
     # Calculate accuracy
    correct_prediction =tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))       
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
 #   for i in range(np.int(test_size/input_size)):# memory can't load all for once
 #           test_err[0]+=1-accuracy.eval(feed_dict={x: X_test[input_size*i:input_size*(i+1),:,:], y: y_test[input_size*i:input_size*(i+1),:]})#!!!!!!!!!
 #   test_err[0]/=test_size/input_size
    test_err[0]=1-accuracy.eval({x: X_test, y: y_test})
    print("epoch[{:3}]".format(0)," test_accurata:",1-test_err[0])
    
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_cost=0.
        total_batch = int(np.shape(X_train)[0]/batch_size)
        #=====shuffle data================
        shuffle_idx=np.random.choice(train_size,size=train_size)
        shuffle_x=X_train[shuffle_idx,:,:]
        shuffle_y=y_train[shuffle_idx,:]
        #=====
        # Loop over all batches
        for i in range(total_batch):
            #================ get batch data========================
            batch_x = shuffle_x[i*batch_size:(i+1)*batch_size,:,:]#
            batch_y = shuffle_y[i*batch_size:(i+1)*batch_size,:] 
            #=======================================================
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y})          
        #=========================== Calculate accuracy #===========================
        correct_prediction =tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))       
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        
    #    for i in range(np.int(train_size/input_size)):# memory can't load all for once
    #        test_err[epoch+1]+=1-accuracy.eval(feed_dict={x: X_test[input_size*i:input_size*(i+1),:,:], y: y_test[input_size*i:input_size*(i+1),:]})#!!!!!!!!!
    #    test_err[epoch+1]/=test_size/input_size
        test_err[epoch+1]=1-accuracy.eval({x: X_test, y: y_test})
        for i in range(np.int(train_size/input_size)):# memory can't load all for once
            train_err[epoch+1]+=1-accuracy.eval(feed_dict={x: X_train[input_size*i:input_size*(i+1),:,:], y: y_train[input_size*i:input_size*(i+1),:]})#!!!!!!!!!
        train_err[epoch+1]/=train_size/input_size
        #train_err[epoch+1]=1-accuracy.eval({x: X_train, y: y_train})
        if epoch % display_step == 0:
            print("epoch[{:3}]".format(epoch+1)," test_accurata:",1-test_err[epoch+1],"train_accurata:",1-train_err[epoch+1])
        #=========================== Calculate accuracy #===========================         
    print("Accuracy:", accuracy.eval({x: X_test, y: y_test}))
WS
tEnd = time.time()
print("time:{:f} min".format((tEnd-tStart)/60))
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(np.arange(1,training_epochs), train_err[1:training_epochs],label='training error')
ax.plot(np.arange(1,training_epochs), test_err[1:training_epochs],label='test error')
ax.legend()
plt.title("learning curve")
plt.xlabel("epoch")
plt.ylabel("error rate")
plt.show()           
          
        