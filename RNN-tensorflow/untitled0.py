# -*- coding: utf-8 -*-
"""
Created on Wed May  3 21:58:01 2017

@author: User
"""

import tensorflow as tf
#from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.rnn import BasicLSTMCell, static_rnn
import numpy as np
#mnist = input_data.read_data_sets('/tmp/data', one_hot = True)

# 來回訓練三次
hm_epochs = 10
# 0~9 十種的多元分類
n_classes = 2
# 因為是要用 SGD 方法，所以定義，在 mnist 55000 training data 中，每 128 組最佳化一次。
batch_size = 200
# 每一次要輸入的 x 的大小
chunk_size = 128 # feature size
# 有幾個要輸入的 x 
n_chunks = 80 # time step
# hidden state size
rnn_size = 128

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

x = tf.placeholder('float',[None, n_chunks, chunk_size])
y = tf.placeholder('float', [None, n_classes])
total_batch = int(np.shape(X_train)[0]/batch_size)
def recurrent_neural_network(x):
	
    layer = {'weights':tf.Variable(tf.random_normal([rnn_size, n_classes])),
			'biases':tf.Variable(tf.random_normal([n_classes]))}
	# transpose 這個 funtion 是對矩陣做 不同維度座標軸的轉換，這邊把一張圖片轉成以每列為單位的輸入
   
    x = tf.transpose(x, [1,0,2])
    x = tf.reshape(x, [-1, chunk_size])

    x = tf.split(axis=0, num_or_size_splits=n_chunks, value=x)
   
	# 定義要被 loop 的基本單元
    lstm_cell = BasicLSTMCell(rnn_size)
	# 選一個把 cell 串起來的 model
    outputs, states = static_rnn(lstm_cell, x, dtype= tf.float32)
	# 用一個 full connection layer 輸出預測
    output = tf.matmul(outputs[-1], layer['weights']) + layer['biases']
    return output

def train_neural_network(x):

	prediction = recurrent_neural_network(x)

	cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))

	optimizer = tf.train.AdamOptimizer().minimize(cost)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for epoch in range(hm_epochs):
                  epoch_loss = 0
                  for i in range(total_batch):
                      epoch_x = X_train[batch_size*i:batch_size(i+1),:,:]
                      epoch_y = y_train[batch_size*i:batch_size(i+1),:,:]
                      epoch_x = epoch_x.reshape((batch_size, n_chunks, chunk_size))
                      _, c = sess.run([optimizer,cost], feed_dict = {x: epoch_x, y: epoch_y})
                      epoch_loss += c
                  print('Epoch completed:',epoch, hm_epochs,'loss:',epoch_loss)
		correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		print ('Accuracy:' ,accuracy.eval({x:X_test, y:y_test}))

train_neural_network(x)