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

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
'''
weights_auto = {           
           'h1_e':tf.Variable(tf.random_normal([784,700])),
           'h2_e':tf.Variable(tf.random_normal([700,400])),
           'h3_e':tf.Variable(tf.random_normal([400,100])),
           'h1_d':tf.Variable(tf.random_normal([100,400])),
           'h2_d':tf.Variable(tf.random_normal([400,700])),
           'h3_d':tf.Variable(tf.random_normal([700,784]))
           }

biases_auto = {
           'b1_e':tf.Variable(tf.random_normal([700])),
           'b2_e':tf.Variable(tf.random_normal([400])),
           'b3_e':tf.Variable(tf.random_normal([100])),
           'b1_d':tf.Variable(tf.random_normal([400])),
           'b2_d':tf.Variable(tf.random_normal([700])),
           'b3_d':tf.Variable(tf.random_normal([784]))
          }

#=========== discriminator's weight and biase ==============================          
weights_disc = {           
           'h1':tf.Variable(tf.random_normal([100,100])),
           'h2':tf.Variable(tf.random_normal([100,50])),
           'h3':tf.Variable(tf.random_normal([50,1])),           
           } 
biases_disc = {
           'b1':tf.Variable(tf.random_normal([100])),
           'b2':tf.Variable(tf.random_normal([50])),
           'b3':tf.Variable(tf.random_normal([1])),          
          }

#================== claim ==============================================
# define placeholder for inputs to network
x_image = tf.placeholder(tf.float32, [None, 28*28])   # 28x28
y = tf.placeholder(tf.float32, [None])   # for discriminator
#keep_prob = tf.placeholder(tf.float32)
sample = tf.placeholder(tf.float32, [None, 100])#store sample from gaussian
#================================== autoencoder ===============================
#================================== (encoder) - generator ===================== 
def encoder(image):
    h1_layer_e = tf.nn.relu(tf.matmul(image,weights_auto['h1_e'])+biases_auto['b1_e'])
    h2_layer_e = tf.nn.relu(tf.matmul(h1_layer_e,weights_auto['h2_e'])+biases_auto['b2_e'])
    z_layer = tf.matmul(h2_layer_e,weights_auto['h3_e'])+biases_auto['b3_e']
    return z_layer
encode = encoder(x_image)
'''
#===== test
h1_layer_e = tf.nn.relu(tf.matmul(x_image,weights_auto['h1_e'])+biases_auto['b1_e'])
h2_layer_e = tf.nn.relu(tf.matmul(h1_layer_e,weights_auto['h2_e'])+biases_auto['b2_e'])
z_layer = tf.nn.relu(tf.matmul(h2_layer_e,weights_auto['h3_e'])+biases_auto['b3_e'])
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(encode, feed_dict={x_image: np.reshape(train_data[0],[1,-1])}))#, feed_dict={x_image: batch_xs, keep_prob: 0.5}
    #print(sess.run(h2_layer_e, feed_dict={x_image: np.reshape(train_data[0],[1,-1])}))#, feed_dict={x_image: batch_xs, keep_prob: 0.5}
#===== test
'''
#=============================== (decoder) ===========================
def decoder(z):# z:encoded x
    h1_layer_d = tf.matmul(z,weights_auto['h1_d'])+biases_auto['b1_d']
    h2_layer_d = tf.nn.relu(tf.matmul(h1_layer_d,weights_auto['h2_d'])+biases_auto['b2_d'])
    output_layer = tf.nn.relu(tf.matmul(h2_layer_d,weights_auto['h3_d'])+biases_auto['b3_d'])
    return output_layer
    
#======================== autoencoder ======================================
def autoencoder(image):
    return decoder(encoder(image))
reconstruct_image = autoencoder(x_image)
#==================== discriminator ===========================================
def discriminator(x):
    h1_layer = tf.nn.relu(tf.matmul(x,weights_disc['h1'])+biases_disc['b1'])
    h2_layer = tf.nn.relu(tf.matmul(h1_layer,weights_disc['h2'])+biases_disc['b2'])
    #output_layer =  tf.nn.softmax(tf.matmul(h2_layer,weights_disc['h3'])+biases_disc['b3'])#!!!!!!!!!
    output_layer =  tf.nn.sigmoid(tf.matmul(h2_layer,weights_disc['h3'])+biases_disc['b3'])#!!!!!!!!!
    #tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output_layer, labels=y)   
    return output_layer
'''
#===== test discriminator output
enc = encoder(x_image)[:2,:10]#discriminator(encoder(x_image))    
disc_fortest = discriminator(encoder(x_image))    

#h1_layer_e = tf.nn.relu(tf.matmul(x_image,weights_auto['h1_e'])+biases_auto['b1_e'])
#h2_layer_e = tf.nn.relu(tf.matmul(h1_layer_e,weights_auto['h2_e'])+biases_auto['b2_e'])
#z_layer = tf.nn.relu(tf.matmul(h2_layer_e,weights_auto['h3_e'])+biases_auto['b3_e'])
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(disc_fortest, feed_dict={x_image:train_data[:5]}))#, feed_dict={x_image: batch_xs, keep_prob: 0.5}
    #print(sess.run(h2_layer_e, feed_dict={x_image: np.reshape(train_data[0],[1,-1])}))#, feed_dict={x_image: batch_xs, keep_prob: 0.5}
#===== test
'''
#==================== (loss) ==================================================
loss_autoencoder = tf.reduce_mean(tf.square(x_image - autoencoder(x_image)))
g_loss = tf.reduce_mean(tf.log((1e-6)+1-discriminator(encoder(x_image))))#for generator
d_loss1 = -tf.reduce_mean(tf.log((1e-6) + discriminator(sample)))
d_loss2 = -g_loss
d_loss = d_loss1 + d_loss2
#==================== (training) ==============================================
train_list_disc = [weights_disc['h1'],weights_disc['h2'],weights_disc['h3'],biases_disc['b1'],biases_disc['b2'],biases_disc['b3']]
train_list_g = [weights_auto['h1_e'],weights_auto['h2_e'],weights_auto['h3_e'],             
                   biases_auto['b1_e'],biases_auto['b2_e'],biases_auto['b3_e']]                 

train_autoencoder = tf.train.AdamOptimizer(1e-4).minimize(loss_autoencoder)
train_encoder = tf.train.AdamOptimizer(1e-4).minimize(g_loss,var_list = train_list_g)
#train_discriminator_part1 = tf.train.AdamOptimizer(1e-4).minimize(d_loss1)
train_discriminator = tf.train.AdamOptimizer(1e-4).minimize(d_loss ,var_list = train_list_disc)

#==============================================================================

'''
#test shape
with tf.Session() as sess:    
    sess.run(tf.global_variables_initializer())    
    print("reconstuct_image's shape:{}".format(np.shape(aaa)))
'''
#===============================================================================
#prediction = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)
tStart = time.clock()#record time
print("test timer:", time.clock() - tStart)


#====================== initial ======================
record_loss_auto = []#np.zeros([1000])
record_loss_gen = []#generator loss
record_loss_disc = []#dicriminator loss
batch_size = 100
train_size = np.shape(train_data)[0]
global_reconstruct_image = np.zeros([10,28,28,3])#for store reconstruct image
total_batch = int(train_size/batch_size)
init = tf.global_variables_initializer()
saver = tf.train.Saver()#for store argument
z_code =  np.zeros([batch_size,100])
z_code_all = np.zeros([train_size,100])
sample_data = np.zeros([batch_size,100])
indices = np.random.choice(a=train_size,size=train_size,replace=False)
#=====================================================
with tf.Session() as sess:
    sess.run(init)	
    if os.path.exists(os.getcwd()+"\\model\\AAE.ckpt"):
        saver.restore(sess, os.getcwd()+"\\model\\AAE.ckpt")
    else:
        print("can't load the argument")
  
    #============= (training) ==========================  
    for epoch in range(200):
        for i in range(total_batch):
            #========= get batch data (shaffle) ==================
            #batch_xs, batch_ys = mnist.train.next_batch(100)          
            batch_xs = train_data[indices][i*batch_size:(i+1)*batch_size] 
            #============= training step  =========================
            '''     
            bbb = sess.run(h_conv2, feed_dict={x_image: np.reshape(train_data[0],[1,28,28,3]), keep_prob: 0.5})
            aaa = sess.run(reconstuct_image, feed_dict={x_image: np.reshape(train_data[0],[1,28,28,3]), keep_prob: 0.5})
            print("h_conv2_d's shape:{}".format(np.shape(bbb)))
            print("reconstuct_image's shape:{}".format(np.shape(aaa)))
            ''' 
            sess.run(train_autoencoder, feed_dict={x_image: batch_xs})            
            z_code = sess.run(encode, feed_dict={x_image: batch_xs})
            sample_data = np.random.multivariate_normal(mean = np.zeros([100]), cov = np.identity(100),size = batch_size)
            sess.run(train_discriminator, feed_dict={x_image: batch_xs,sample:sample_data})    
            sess.run(train_encoder, feed_dict={x_image: batch_xs})    
            
        #================ (evaluate loss) =====================================          
        record_loss_auto.append(sess.run(loss_autoencoder, feed_dict={x_image: train_data[:batch_size],sample:sample_data}))          
        record_loss_gen.append(sess.run(-g_loss, feed_dict={x_image: train_data[:batch_size],sample:sample_data}))          
        record_loss_disc.append(sess.run(d_loss, feed_dict={x_image: train_data[:batch_size],sample:sample_data}))          
        #if i % 100 == 0:        
    #    print("disc===:",sess.run(disc_fortest, feed_dict={x_image: train_data,sample:sample_data, keep_prob: 0.5})[0])
        print("epoch:{0:4d}  Loss_auto:{1:.3f}  Loss_gen:{2:.3f}  Loss_disc:{3:.3f}".format(epoch+1,
              record_loss_auto[-1],
                record_loss_gen[-1],
                record_loss_disc[-1]))
    #================================================================  
    z_code_all = sess.run(encode, feed_dict={x_image: train_data,sample:np.zeros([1,100]) })#get all image's z code
    '''
    from scipy.misc import toimage
    test = sess.run(reconstuct_image, feed_dict={x_image: train_data[:10]/255, keep_prob: 0.5})         
    #plot image for test    
    h= w = 28
    k = 0
    for i in range(k,k+5):
        plt.subplot(2,5,(i-k)%9+1)# create a grid of 3x3 images
        plt.imshow(toimage([test[i,:h,:w,0],test[i,:h,:w,1],test[i,:h,:w,2]]))#plt.imshow(toimage([tB[i,:h,:w,0],tB[i,:h,:w,1],tB[i,:h,:w,2]]))    
        '''  
    global_reconstruct_image = sess.run(reconstruct_image, feed_dict={x_image: train_data[:10] })
    save_path = saver.save(sess, os.getcwd()+"\\model\\AAE.ckpt")#save augument 

tEnd = time.clock()#record time
t_time =  tEnd - tStart
print("total time:{0:.3f} min.".format(t_time/60))
#===plot data =====================
fig = plt.figure()

ax = fig.add_subplot(1,1,1)
ax.plot(record_loss_auto,label='loss_auto')
#ax.plot(train_err,label='train error')
ax.legend()
plt.xlabel("epoch")

fig2 = plt.figure()
ax2 = fig2.add_subplot(1,1,1)
ax2.plot(record_loss_gen,label='loss_gen')
ax2.plot(record_loss_disc,label='loss_disc')
ax2.legend()
plt.title("learning curve")
plt.xlabel("epoch")
plt.ylabel("cost")
plt.ion()
plt.show()  
#============ (plot z code in 2D) =======================
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
'''
z is the output of encoder
z = model.encoding(sess, data)
'''

tsne = TSNE(n_components = 2, random_state = 0)
t_z = tsne.fit_transform(z_code_all)
np.shape(z_code_all)

'''
plot the t_z, color is determined by label
'''
colors = plt.cm.rainbow(np.linspace(0, 1, 10))
scatter = []
index = range(10)
for i in range(10):
    tmp = np.where(train_label == i)
    scatter.append(plt.scatter(t_z[tmp, 0], t_z[tmp, 1], c = colors[i] ,s = 10))

plt.legend(scatter, index)
plt.show()
#===================================
#===plot image =====================
global_reconstruct_image2 = np.reshape(global_reconstruct_image,[-1,28,28])
np.shape(global_reconstruct_image2)
from scipy.misc import toimage
#X_train=np.reshape(data_sets['images_test'], (-1,32, 32,3))
h= w = 28
k = 0
for i in range(k,k+5):
    plt.subplot(2,5,i%9+1)# create a grid of 3x3 images
    plt.imshow(toimage([global_reconstruct_image2[i,:h,:w],global_reconstruct_image2[i,:h,:w],global_reconstruct_image2[i,:h,:w]]))#plt.imshow(toimage([tB[i,:h,:w,0],tB[i,:h,:w,1],tB[i,:h,:w,2]]))    
#print(np.shape(data_sets['images_test']))
data = np.reshape(train_data[:10],[-1,28,28])
for i in range(5):
    plt.subplot(2,5,i+1+5)   
    plt.imshow(toimage([data[i,:h,:w],data[i,:h,:w],data[i,:h,:w]]))
np.shape(data)