import tensorflow as tf
import random
import numpy as np
import matplotlib.pyplot as plt
import datetime
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('data/mnist/', one_hot=True)

tf.reset_default_graph() 
sess = tf.InteractiveSession()
x = tf.placeholder("float", shape = [None, 28,28,1]) #shape in CNNs is always None x height x width x color channels
y_ = tf.placeholder("float", shape = [None, 10]) #shape is always None x number of classes



def Conv2d_Layer(inChannel, outChannel,x):
    W_conv = tf.Variable(tf.truncated_normal([3, 3, inChannel, outChannel], stddev=0.1))#shape is filter x filter x input channels x output channels
    b_conv = tf.Variable(tf.constant(0.1, shape = [outChannel])) #shape of the bias just has to match output channels of the filter
    h_conv = tf.nn.conv2d(input=x, filter=W_conv, strides=[1, 1, 1, 1], padding='SAME') + b_conv
    h_conv = tf.nn.relu(h_conv)
    h_pool = tf.nn.max_pool(h_conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    return h_pool

def FullConn_Layer(x, leftNum,rightNum):
    w = tf.Variable(tf.truncated_normal([leftNum, rightNum], stddev=0.1))
    b = tf.Variable(tf.constant(.1, shape = [rightNum]))
    h_flat = tf.reshape(x, [-1, leftNum])
    h_fc = tf.matmul(h_flat, w) + b
    return h_fc

h_pool1 = Conv2d_Layer(1,32,x)
h_pool2 = Conv2d_Layer(32,64,h_pool1)
h_pool3 = Conv2d_Layer(64,128,h_pool2)
h_fc1 = FullConn_Layer(h_pool3,4*4*128,1024)

#Dropout Layer
fc1_relu = tf.nn.relu(h_fc1)
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(fc1_relu, keep_prob)

y = FullConn_Layer(h_fc1_drop,1024,10)

crossEntropyLoss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = y))
trainStep = tf.train.AdamOptimizer().minimize(crossEntropyLoss)
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess.run(tf.global_variables_initializer())
batchSize = 50
for i in range(10000):
    batch = mnist.train.next_batch(batchSize)
    trainingInputs = batch[0].reshape([batchSize,28,28,1])
    trainingLabels = batch[1]
    if i%100 == 0:
        trainAccuracy = accuracy.eval(session=sess, feed_dict={x:trainingInputs, y_: trainingLabels, keep_prob: 1.0})
        print ("step %d, training accuracy %g"%(i, trainAccuracy))
    trainStep.run(session=sess, feed_dict={x: trainingInputs, y_: trainingLabels, keep_prob: 0.5})

testInputs = mnist.test.images
testInputs = testInputs.reshape([10000,28,28,1])
testLabels = mnist.test.labels
acc = accuracy.eval(session=sess, feed_dict = {x: testInputs, y_: testLabels, keep_prob: 1.0})
print("testing accuracy: {}".format(acc))
print("testing accuracy: {}".format(acc))
