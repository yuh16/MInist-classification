#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 15:08:09 2017

@author: yuhan
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import argparse
import tensorflow as tf
import numpy as np
import pandas
import random
import matplotlib.pyplot as plt

IMAGE_HEIGHT  = 28
IMAGE_WIDTH   = 28
NUM_CHANNELS  = 3
BATCH_SIZE    = 50

VALIDATION_SIZE = 3000

#dataset_path
train_labels_file = 'train.csv'
test_file = 'test.csv'


def encode_label(label):
  return int(label)

def read_label_file(file):
    f = open(file,'r')
    trainingdatas = []
    labels = []
    with open(train_labels_file,'r') as f:
        next(f)
        for line in f:
            trainingdata = line.split(",")
            trainingdatas.append(trainingdata[1:])
            labels.append(encode_label(trainingdata[0]))
    return trainingdatas, labels

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
  
  
def next_batch(batch_size):    
    global train_images
    global train_labels
    global index_in_epoch
    global epochs_completed
    
    start = index_in_epoch
    index_in_epoch += batch_size
    
    # when all trainig data have been already used, it is reorder randomly    
    if index_in_epoch > num_examples:
        # finished epoch
        epochs_completed += 1
        # shuffle the data
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        train_images = train_images[perm]
        train_labels = train_labels[perm]
        # start next epoch
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch
    return train_images[start:end], train_labels[start:end]

def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


#train_filepaths, train_labels = read_label_file(train_labels_file)
#test_filepaths, test_labels = read_label_file(test_file)


data = pandas.read_csv(train_labels_file)

labels_flat = data[[0]].values.ravel()
labels_count = np.unique(labels_flat).shape[0]
labels = dense_to_one_hot(labels_flat, labels_count)
labels = labels.astype(np.uint8)

train_data = data.iloc[:,1:].values
train_data = train_data.astype(np.float)

test_data = pandas.read_csv(test_file).values
test_data = test_data.astype(np.float)
test_size, im_size = test_data.shape


train_data = np.multiply(train_data, 1.0 / 255.0)
test_data = np.multiply(test_data, 1.0 / 255.0)
image_width = image_height = np.sqrt(im_size).astype(np.uint8)

validation_images = train_data[:VALIDATION_SIZE]
validation_labels = labels[:VALIDATION_SIZE]

train_images = train_data[VALIDATION_SIZE:]
train_labels = labels[VALIDATION_SIZE:]



x = tf.placeholder('float', shape=[None, im_size])       # mnist data image of shape 28*28=784
y_ = tf.placeholder('float', shape=[None, labels_count])    # 0-9 digits recognition => 10 classes
keep_prob = tf.placeholder(tf.float32)




W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1,28,28,1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
     
                  

#drop_conv = tf.placeholder('float')
#drop_hidden = tf.placeholder('float')                  
                  
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
predict = tf.argmax(y_conv, 1)
init = tf.initialize_all_variables()
sess = tf.InteractiveSession()
sess.run(init)

epochs_completed = 0
index_in_epoch = 0
num_examples = train_images.shape[0]

for i in range(5000):
  #batch = mnist.train.next_batch(50)
  batch_x, batch_y = next_batch(BATCH_SIZE)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch_x, y_: batch_y, keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x:batch_x, y_: batch_y, keep_prob: 0.5})

#print("test accuracy %g"%accuracy.eval(feed_dict={
#    x: test_data, keep_prob: 1.0}))
print("test accuracy \n")       
predicted_lables = np.zeros(test_data.shape[0])
predicted_lables = predict.eval(feed_dict={
    x: test_data, keep_prob: 1.0})

           
# save results
np.savetxt('submission.csv', 
           np.c_[range(1,len(test_data)+1),predicted_lables], 
           delimiter=',', 
           header = 'imageid,label', 
           comments = '', 
           fmt='%d')

sess.close()




