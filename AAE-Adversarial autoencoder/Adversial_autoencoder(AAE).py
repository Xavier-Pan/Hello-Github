# -*- coding: utf-8 -*-
"""
Created on Wed May 31 22:03:04 2017

@author: SY-Pan
"""
"""
Created on Thu Apr 13 10:16:16 2017

"""
#import read_MNIST_M
import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
import os
if not os.path.exists(os.getcwd()+"\\model"):   
    os.mkdir(os.getcwd()+"\\model")

#========================= load data =======================================
train_data = np.load(r'C:\Users\User\Desktop\Deep Learning\HW4\data.npy')
train_label = np.load(r'C:\Users\User\Desktop\Deep Learning\HW4\label.npy')     
#===========================================================================
'''
def compute_accuracy(v_xs):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, keep_prob: 1})
    return result
'''
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

weights_auto = {           
           'h1_e':tf.Variable(tf.random_normal([784,700])),
           'h2_e':tf.Variable(tf.random_normal([700,400])),
           'h3_e':tf.Variable(tf.random_normal([400,100])),
           'h1_d':tf.Variable(tf.random_normal([100,400])),
           'h2_d':tf.Variable(tf.random_normal([400,700])),
           'h3_d':tf.Variable(tf.random_normal([700,784]))
           }
 
biases_auto = {
           'b1_e':tf.Variable(tf.random_normal(700)),
           'b2_e':tf.Variable(tf.random_normal(400)),
           'b3_e':tf.Variable(tf.random_normal(100)),
           'b1_d':tf.Variable(tf.random_normal(400)),
           'b2_d':tf.Variable(tf.random_normal(700)),
           'b3_d':tf.Variable(tf.random_normal(784))
          }
#=========== discriminator's weight and biase ==============================          
weights_disc = {           
           'h1':tf.Variable(tf.random_normal([784,700])),
           'h2':tf.Variable(tf.random_normal([700,400])),
           'h3':tf.Variable(tf.random_normal([400,1])),           
           } 
biases_disc = {
           'b1':tf.Variable(tf.random_normal(700)),
           'b2':tf.Variable(tf.random_normal(400)),
           'b3':tf.Variable(tf.random_normal(1)),          
          }
#==========================================================================
# define placeholder for inputs to network
x_image = tf.placeholder(tf.float32, [None, 28*28])   # 28x28
y = tf.placeholder(tf.float32, [None])   # for discriminator
keep_prob = tf.placeholder(tf.float32)
#x_image = #tf.reshape(xs, [-1, 28, 28, 3])
# print(x_image.shape)  # [n_samples, 28,28,1]
h1_kernel_num = 16
h1_kernel_size = 5
h2_kernel_num = 32

#================================== autoencoder ===============================
#================================== encoder ===============================
h1_layer_e = tf.nn.relu(tf.matmul(x_image,weights_auto['h1_e'])+biases_auto['b1_e'])
h2_layer_e = tf.nn.relu(tf.matmul(h1_layer_e,weights_auto['h2_e'])+biases_auto['b2_e'])
z_layer = tf.nn.relu(tf.matmul(h2_layer_e,weights_auto['h3_e'])+biases_auto['b3_e'])

#=============================== generator(decoder) ===========================
h1_layer_d = tf.nn.relu(tf.matmul(z_layer,weights_auto['h1_d'])+biases_auto['b1_d'])
h2_layer_d = tf.nn.relu(tf.matmul(h1_layer_d,weights_auto['h2_d'])+biases_auto['b2_d'])
output_layer = tf.nn.relu(tf.matmul(h2_layer_d,weights_auto['h3_d'])+biases_auto['b3_d'])

#==============================================================================
loss_autoencoder = tf.reduce_mean(tf.square(x_image - output_layer))
loss_autoencoder = tf.reduce_mean(tf.square(x_image - output_layer)) 

#==================== discriminator ===========================================================
def discriminator():
    h1_layer = tf.nn.relu(tf.matmul(x_image,weights_auto['h1'])+biases_auto['b1'])
    h2_layer = tf.nn.relu(tf.matmul(h1_layer,weights_auto['h2'])+biases_auto['b2'])
    output_layer =  tf.nn.softmax(tf.matmul(h2_layer,weights_auto['h3'])+biases_auto['b3'])#!!!!!!!!!
    #tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output_layer, labels=y)   
    return output_layer
#===============================================================================
#reconstuct_image = h_conv1_d#tf.reshape(h_conv1_d,[-1, 28*28*3])
'''
#test shape
with tf.Session() as sess:    
    sess.run(tf.global_variables_initializer())
    bbb = sess.run(h_conv2, feed_dict={x_image: np.reshape(train_data[0],[1,28,28,3]), keep_prob: 0.5})
    aaa = sess.run(h_conv2_d, feed_dict={x_image: np.reshape(train_data[0],[1,28,28,3]), keep_prob: 0.5})
    print("h_conv2_d's shape:{}".format(np.shape(bbb)))
    print("reconstuct_image's shape:{}".format(np.shape(aaa)))
'''
#===============================================================================
#prediction = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)
tStart = time.clock()#record time
print("test timer:", time.clock() - tStart)
# the error between prediction and real data
loss = tf.reduce_mean(tf.square(x_image - reconstuct_image))       # loss
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)#.maximize(loss)
#====================== initial ======================
record_loss = []#np.zeros([1000])
batch_size = 100
train_size = np.shape(train_data)[0]
test_size = np.shape(test_data)[0]
global_reconstruct_image = np.zeros([10,28,28,3])#for store reconstruct image
total_batch = int(train_size/batch_size)
init = tf.global_variables_initializer()
saver = tf.train.Saver()#for store argument

#=====================================================
with tf.Session() as sess:
    sess.run(init)	
    if os.path.exists(os.getcwd()+"\\model\\autoEncoder2.ckpt"):
        saver.restore(sess, os.getcwd()+"\\model\\autoEncoder2.ckpt")
    else:
        print("can't load the argument")
       
      
    for epoch in range(10):
        for i in range(total_batch):
            #========= batch data ==================
            #batch_xs, batch_ys = mnist.train.next_batch(100)
            #train_data, train_label, valid_data,valid_label, test_data,test_label
            batch_xs = train_data[i*batch_size:(i+1)*batch_size,:,:,:]/255
            #=======================================
            '''     
            bbb = sess.run(h_conv2, feed_dict={x_image: np.reshape(train_data[0],[1,28,28,3]), keep_prob: 0.5})
            aaa = sess.run(reconstuct_image, feed_dict={x_image: np.reshape(train_data[0],[1,28,28,3]), keep_prob: 0.5})
            print("h_conv2_d's shape:{}".format(np.shape(bbb)))
            print("reconstuct_image's shape:{}".format(np.shape(aaa)))
            '''
            sess.run(train_step, feed_dict={x_image: batch_xs, keep_prob: 0.5})            
        record_loss.append(sess.run(loss, feed_dict={x_image: batch_xs, keep_prob: 0.5}))          
        #if i % 100 == 0:        
        print("epoch:{0:4d}  Loss:{1:.3f}".format(epoch+1, record_loss[-1]))
        from scipy.misc import toimage
        test = sess.run(reconstuct_image, feed_dict={x_image: train_data[:10]/255, keep_prob: 0.5}) 
        '''
        #plot image for test    
        h= w = 28
        k = 0
        for i in range(k,k+5):
            plt.subplot(2,5,(i-k)%9+1)# create a grid of 3x3 images
            plt.imshow(toimage([test[i,:h,:w,0],test[i,:h,:w,1],test[i,:h,:w,2]]))#plt.imshow(toimage([tB[i,:h,:w,0],tB[i,:h,:w,1],tB[i,:h,:w,2]]))    
        '''  
    global_reconstruct_image = sess.run(reconstuct_image, feed_dict={x_image: train_data[:10], keep_prob: 0.5})
    save_path = saver.save(sess, os.getcwd()+"\\model\\autoEncoder2.ckpt")#save augument 

tEnd = time.clock()#record time
t_time =  tEnd - tStart
print("total time:{0:.3f} min.".format(t_time/60))
#===plot data =====================
fig = plt.figure()
'''
ax = fig.add_subplot(1,1,1)
ax.plot(record_loss,label='test error')
#ax.plot(train_err,label='train error')
ax.legend()
plt.xlabel("epoch")
'''
fig2 = plt.figure()
ax2 = fig2.add_subplot(1,1,1)
ax2.plot(record_loss,label='loss')
ax2.legend()
plt.title("learning curve")
plt.xlabel("epoch")
plt.ylabel("cost")
plt.ion()
plt.show()  
#===================================
#===plot image =====================
from scipy.misc import toimage
#X_train=np.reshape(data_sets['images_test'], (-1,32, 32,3))
h= w = 28
k = 0
for i in range(k,k+5):
    plt.subplot(2,5,(i-k)%9+1)# create a grid of 3x3 images
    plt.imshow(toimage([global_reconstruct_image[i,:h,:w,0],global_reconstruct_image[i,:h,:w,1],global_reconstruct_image[i,:h,:w,2]]))#plt.imshow(toimage([tB[i,:h,:w,0],tB[i,:h,:w,1],tB[i,:h,:w,2]]))    
#print(np.shape(data_sets['images_test']))

for i in range(5):
    plt.subplot(2,5,i+1+5)   
    plt.imshow(toimage([train_data[i,:h,:w,0],train_data[i,:h,:w,1],train_data[i,:h,:w,2]]))


