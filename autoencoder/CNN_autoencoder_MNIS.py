# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 10:16:16 2017

@author: User
"""
#import read_MNIST_M
import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
#from tensorflow.examples.tutorials.mnist import input_data
# number 1 to 10 data
#mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
def load_data():
    MNIST_M = np.load(r'C:\Users\User\Desktop\Deep Learning\HW4\Mnist_M.npy')
    train_data, train_label = MNIST_M[0]
    valid_data, valid_label = MNIST_M[1]
    test_data, test_label = MNIST_M[2]    
    return train_data, train_label, valid_data,valid_label, test_data,test_label
  

train_data, train_label, valid_data,valid_label, test_data,test_label = load_data()

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

def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# define placeholder for inputs to network
x_image = tf.placeholder(tf.float32, [None, 28,28,3])/255   # 28x28
keep_prob = tf.placeholder(tf.float32)
#x_image = #tf.reshape(xs, [-1, 28, 28, 3])
# print(x_image.shape)  # [n_samples, 28,28,1]
#===============================================================================
## conv1 layer ##
W_conv1 = weight_variable([5,5, 3,32]) # patch 5x5, in size 1, out size 32
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # output size 24x24x32


## conv2 layer ##
W_conv2 = weight_variable([5,5, 32, 64]) # patch 5x5, in size 32, out size 64
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2) # output size 20x20x64

## fc1 layer ##
W_fc1 = weight_variable([20*20*64, 1024])
b_fc1 = bias_variable([1024])
# [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
h_pool2_flat = tf.reshape(h_conv2, [-1, 20*20*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
#h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

## fc2 layer ##
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
out_put = tf.matmul(h_fc1, W_fc2) + b_fc2

#=========reconstruct
b_fc2_d = bias_variable([1024])
W_fc2_d = weight_variable([10, 1024])
h_fc2_d = tf.matmul(out_put, W_fc2_d) + b_fc2_d

W_fc1_d = weight_variable([1024,20*20*64])
b_fc1_d = bias_variable([20*20*64])
h_fc1_d = tf.matmul(h_fc2_d, W_fc1_d) + b_fc1_d

h2_flat_d = tf.reshape(h_fc1_d, [-1, 20,20,64])
# [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
W_conv2 = weight_variable([5,5, 64, 32]) # patch 5x5, in size 32, out size 64
b_conv2 = bias_variable([32])

h_conv2_d =  tf.layers.conv2d_transpose(
    inputs = h2_flat_d,
    filters = 32,
    kernel_size = [5,5],
    strides=[1,1],
    activation=tf.nn.relu
)

W_conv1 = weight_variable([5,5, 32,3]) # patch 5x5, in size 1, out size 32
b_conv1 = bias_variable([3])
h_conv1_d =  tf.layers.conv2d_transpose(
    h_conv2_d,
    3,
    [5,5],
    strides=[1,1],
    padding='VALID',
    data_format='channels_last',
    activation=tf.nn.relu,
    use_bias=True,
    kernel_initializer=None,
    bias_initializer=tf.zeros_initializer(),
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    trainable=True,
    name=None,
    reuse=None)

reconstuct_image = h_conv1_d#tf.reshape(h_conv1_d,[-1, 28*28*3])
#===============================================================================
prediction = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)


# the error between prediction and real data
loss = tf.reduce_mean(tf.sqrt(x_image - reconstuct_image))       # loss
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)	
    print('正常')
    # important step
    # tf.initialize_all_variables() no long valid from
    # 2017-03-02 if using tensorflow >= 0.12
    record_loss = np.zeros([1000])
    batch_size = 200
    train_size = np.shape(train_data)[0]
    test_size = np.shape(test_data)[0]
    total_batch = int(train_size/batch_size)
    for epoch in range(50):
    for i in range(total_batch):
        #========= batch data ==================
        #batch_xs, batch_ys = mnist.train.next_batch(100)
        #train_data, train_label, valid_data,valid_label, test_data,test_label
        batch_xs = train_data[i*batch_size:(i+1)*batch_size]
        #=======================================
        sess.run(train_step, feed_dict={x_image: batch_xs, keep_prob: 0.5})
        record_loss[i] = sess.run(loss, feed_dict={x_image: batch_xs, keep_prob: 0.5})
        if i % 50 == 0:
            print(record_loss[i])
        
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
plt.xlabel("batch iteration")
plt.ylabel("error rate")
plt.ion()
plt.show()  