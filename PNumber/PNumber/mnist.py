import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

print ("下载中。。。")
mnist = input_data.read_data_sets('data/mnist/',one_hot=True)
print("类型是：%s"%(type(mnist)))
print("训练数据量：%d"%(mnist.train.num_examples))
print("测试数据量：%d"%(mnist.test.num_examples))

trainimg = mnist.train.images
trainlabel = mnist.train.labels
testimg = mnist.test.images
testlabel = mnist.test.labels

print("数据类型是%s"%(type(trainimg)))
print("标签类型是%s"%(type(trainlabel)))
print("训练集的shape是%s"%(trainimg.shape,))
print("训练集的标签的shape是%s"%(trainlabel.shape,))
print("测试集的shape是%s"%(testimg.shape,))
print("测试集的标签shape是%s"%(testlabel.shape,))

# 看看数据的样子
nsample = 1
randidx = np.random.randint(trainimg.shape[0],size=nsample)

for i in randidx:
    curr_img = np.reshape(trainimg[i,:],(28,28))
    curr_label = np.argmax(trainlabel[i,:])
    print(trainlabel[i,:])
    plt.matshow(curr_img,cmap=plt.get_cmap('gray'))
    print(""+str(i)+"th 训练数据"+"标签是"+str(curr_label))
    plt.show()


# batch数据
print("batch Learning")
batch_size = 100
batch_xs , batch_ys = mnist.train.next_batch(batch_size)
print("Batch数据 %s"%(type(batch_xs)))
print("Batch标签 %s"%(type(batch_ys)))
print("Batch数据的shape %s"%(batch_xs.shape,))
print("Batch标签的shape %s"%(batch_ys.shape,))

# 设置参数
numClasses =10
inputsize = 784
trainingIterations = 50000
batchSize = 128

X = tf.placeholder(tf.float32,shape=[None,inputsize])
Y = tf.placeholder(tf.float32,shape=[None,numClasses])

W1= tf.Variable(tf.random_normal([inputsize,numClasses],stddev=0.1))
B1 = tf.Variable(tf.constant(0.1),[numClasses])
y_pred = tf.nn.softmax(tf.matmul(X,W1)+B1)

loss = tf.reduce_mean(tf.square(y_pred-Y))
opt = tf.train.GradientDescentOptimizer(learning_rate =0.05).minimize(loss)

corrent_prediction = tf.equal(tf.argmax(y_pred,1),tf.argmax(y_pred,1))
accuracy = tf.reduce_mean(tf.cast(corrent_prediction,"float"))

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(trainingIterations):
    batch = mnist.train.next_batch(batchSize)
    batchInput = batch[0]
    batchLabels = batch[1]
    trainingLoss = sess.run([opt,loss],feed_dict={X:batchInput,Y:batchLabels})
    if i%1000 ==0:
        train_accuracy = accuracy.eval(session=sess,feed_dict={X:batchInput,Y:batchLabels})
        print("step ",i,"jingdu",train_accuracy)
