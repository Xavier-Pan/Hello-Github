'''
A Recurrent Neural Network (LSTM) implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits (http://yann.lecun.com/exdb/mnist/)
Long Short Term Memory paper: http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf

Author: Aymeric Damien
'''

from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
# Import MNIST data
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

'''
To classify images using a recurrent neural network, we consider every image
row as a sequence of pixels. Because MNIST image shape is 28*28px, we will then
handle 28 sequences of 28 steps for every sample.
'''

# Parameters
learning_rate = 0.001
training_iters = 100000
batch_size = 100
display_step = 10
imdb = np.load('imdb_word_emb.npz')
X_train = imdb['X_train']
y_train_ = imdb['y_train']
X_test  = imdb['X_test']
y_test_  = imdb['y_test']
test_size=np.shape(y_test_)[0]
train_size=np.shape(y_train_)[0]
y_train=np.zeros([train_size,2])
y_test=np.zeros([test_size,2])

# Network Parameters
n_input = 128 # MNIST data input (img shape: 28*28)
n_steps = 80 # timesteps
n_hidden = 64 # hidden layer num of features
n_classes = 2 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, n_steps, 1)
    
    # Define a lstm cell with tensorflow
#    tf.get_variable_scope().reuse_variables()
    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
        lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    
    # Get lstm cell output
        outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = RNN(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
with tf.variable_scope(tf.get_variable_scope(), reuse=False):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()
total_batch=int(25000/batch_size)
# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    print ('Accuracy:' ,accuracy.eval({x:X_test, y:y_test}))
    # Keep training until reach max iterations
    epoch_loss = 0
    for i in range(total_batch):
        epoch_x = X_train[batch_size*i:batch_size*(i+1),:,:]
        epoch_y = y_train[batch_size*i:batch_size*(i+1),:]
        epoch_x = epoch_x.reshape((batch_size, 80, 128))
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            _, c = sess.run([optimizer,cost], feed_dict = {x: epoch_x, y: epoch_y})
        epoch_loss += c
        # Run optimization op (backprop)     
        if i % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: epoch_x, y: epoch_y})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: epoch_x, y: epoch_y})
            print("Iter " + str(i) + ", Minibatch Loss= " + \
                  "{:}".format(loss) + ", Training Accuracy= " + \
                  "{:}".format(acc))
      
    print("Optimization Finished!")
    print ('Accuracy:' ,accuracy.eval({x:X_test, y:y_test}))
    # Calculate accuracy for 128 mnist test images
