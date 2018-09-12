from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets('data/mnist/', one_hot=True)


# 设置参数
numClasses = 10
inputSize = 784  
trainingIterations = 50000
batchSize = 64

X = tf.placeholder(tf.float32, shape = [None, inputSize])
y = tf.placeholder(tf.float32, shape = [None, numClasses])

W1 = tf.Variable(tf.random_normal([inputSize, numClasses], stddev=0.1))
B1 = tf.Variable(tf.constant(0.1), [numClasses])

y_pred = tf.nn.softmax(tf.matmul(X, W1) + B1)
loss = tf.reduce_mean(tf.square(y - y_pred))
opt = tf.train.GradientDescentOptimizer(learning_rate = 0.05).minimize(loss)

correct_prediction = tf.equal(tf.argmax(y_pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


config = tf.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.Session(config=config)
init = tf.global_variables_initializer()
sess.run(init)

for i in range(trainingIterations):
    batch = mnist.train.next_batch(batchSize)
    batchInput = batch[0]
    batchLabels = batch[1]
    trainingLoss = sess.run([opt, loss], feed_dict={X: batchInput, y: batchLabels})
    if i%1000 == 0:
        train_accuracy = accuracy.eval(session=sess, feed_dict={X: batchInput, y: batchLabels})
        print ("step %d, training accuracy %g"%(i, train_accuracy))
        batch = mnist.test.next_batch(1000)
        testAccuracy = sess.run(accuracy, feed_dict={X: batch[0], y: batch[1]})
        print ("test accuracy %g"%(testAccuracy))



