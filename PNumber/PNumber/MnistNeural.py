from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets('data/mnist/', one_hot=True)


trainingIterations = 10000 
batchSize = 100 

X = tf.placeholder(tf.float32, shape = [None, 784])
y = tf.placeholder(tf.float32, shape = [None, 10])

def CreateHiddenLayer(X,inputSize,hiddenUnits):
    W = tf.Variable(tf.truncated_normal([inputSize, hiddenUnits], stddev=0.1))
    B = tf.Variable(tf.constant(0.1), [hiddenUnits])
    hiddenLayerOutput = tf.matmul(X, W) + B
    hiddenLayerOutput = tf.nn.relu(hiddenLayerOutput)
    return hiddenLayerOutput

X1 = CreateHiddenLayer(X,784,400)
X2 = CreateHiddenLayer(X1,400,200)
X3 = CreateHiddenLayer(X2,200,100)
X4 = CreateHiddenLayer(X3,100,50)
X5 = CreateHiddenLayer(X4,50,10)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = X5))
opt = tf.train.AdamOptimizer().minimize(loss)

correct_prediction = tf.equal(tf.argmax(X5,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(trainingIterations):
    batch = mnist.train.next_batch(batchSize)
    batchInput = batch[0]
    batchLabels = batch[1]
    trainingLoss = sess.run([opt, loss], feed_dict={X: batchInput, y: batchLabels})
    if i%1000 == 0:
        trainAccuracy = accuracy.eval(session=sess, feed_dict={X: batchInput, y: batchLabels})
        print ("step %d, training accuracy %g"%(i, trainAccuracy))

testInputs = mnist.test.images
testLabels = mnist.test.labels
acc = accuracy.eval(session=sess, feed_dict = {X: testInputs, y: testLabels})
print("testing accuracy: {}".format(acc))