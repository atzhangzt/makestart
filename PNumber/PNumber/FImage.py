import tensorflow as tf 
import numpy as np
import os,glob,cv2
import sys,argparse

def getImage(path,image_size):
    images = []
    image = cv2.imread(path)
    image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
    images.append(image)
    images = np.array(images, dtype=np.uint8)
    images = images.astype('float32')
    images = np.multiply(images, 1.0/255.0)
    return images
def getImages(pathes,image_size):
    images = []
    for path in pathes:
        image = cv2.imread(path)
        image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
        images.append(image)
    images = np.array(images, dtype=np.uint8)
    images = images.astype('float32')
    images = np.multiply(images, 1.0/255.0)
    return images
def getResult(sess,images,img_size,num_channels):
    x_batch = images.reshape(1, img_size,img_size,num_channels)

    y_pred = sess.graph.get_tensor_by_name("y_pred:0")
    x= sess.graph.get_tensor_by_name("x:0")
    y_true = sess.graph.get_tensor_by_name("y_true:0") 
    y_test_images = np.zeros((1, 2)) 
    feed_dict_testing = {x: x_batch, y_true: y_test_images}
    return sess.run(y_pred, feed_dict=feed_dict_testing)

