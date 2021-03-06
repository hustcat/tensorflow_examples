__author__ = 'chapter'

from datetime import datetime
import time

import argparse
import sys

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def weight_varible(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

FLAGS = None


def model(x_image, W1, b1, W2, b2, Wc1, bc1, keep_prob, Wc2, bc2):
    with tf.device('/gpu:0'):

      # layer-1
      h_conv1 = tf.nn.relu(conv2d(x_image, W1) + b1)
      h_pool1 = max_pool_2x2(h_conv1)

      # layer-2
      h_conv2 = tf.nn.relu(conv2d(h_pool1, W2) + b2)
      h_pool2 = max_pool_2x2(h_conv2)

      # full connection
      h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
      h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, Wc1) + bc1)

      # dropout
      h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

      # output layer: softmax
      Wc2 = weight_varible([1024, 10])
      bc2 = bias_variable([10])
      y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, Wc2) + bc2)
      return y_conv

def main(_):
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    print("Download Done!")
      
    # input
    x = tf.placeholder(tf.float32, [None, 784])
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    # conv layer-1
    W1 = weight_varible([5, 5, 1, 32])
    b1 = bias_variable([32])
    
    # conv layer-2
    W2 = weight_varible([5, 5, 32, 64])
    b2 = bias_variable([64])
    
    # full connection
    W_fc1 = weight_varible([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    # dropout
    keep_prob = tf.placeholder(tf.float32)
    
    # output layer: softmax
    W_fc2 = weight_varible([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = model(x_image, W1, b1, W2, b2, W_fc1, b_fc1, keep_prob, W_fc2, b_fc2) 
    y_ = tf.placeholder(tf.float32, [None, 10])
    
    sess = tf.InteractiveSession(config=tf.ConfigProto(
        log_device_placement=True)) 

    # model training
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.arg_max(y_conv, 1), tf.arg_max(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess.run(tf.global_variables_initializer())

    start_time = time.time()
    for i in range(20000):
        batch = mnist.train.next_batch(50)

        if i % 100 == 0:
            train_accuacy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g"%(i, train_accuacy))
        train_step.run(feed_dict = {x: batch[0], y_: batch[1], keep_prob: 0.5})

    duration = time.time() - start_time
    # accuacy on test
    print("test accuracy %g, used %d seconds"%(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}), duration))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
