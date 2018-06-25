#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 3/8/18 11:28 AM
# @Author  : Xin He

import tensorflow as tf

from numpy.random import RandomState

# define training data batch size
batch_size = 8

# define network parameters
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

# input x and y_
x = tf.placeholder(tf.float32, shape=(None, 2), name='x-input')
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')

# define forward progress
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# define cross entropy and back propagation
y = tf.sigmoid(y)
cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)) +
                                (1 - y) *
                                tf.log(tf.clip_by_value(1-y, 1e-10, 1.0)))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

# generate random data
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)
Y = [[int(x1+x2 < 1)] for (x1, x2) in X]

# create a session
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    # initial variables
    sess.run(init_op)

    # show the network parameters before training
    print sess.run(w1)
    print sess.run(w2)

    # training times
    steps = 5000
    for i in xrange(steps):
        start = (i * batch_size) % dataset_size
        end = min(start+batch_size, dataset_size)

        # use sample data to train network and update parameters
        sess.run(train_step,
                 feed_dict={x: X[start:end], y_: Y[start:end]})
        if i % 1000 == 0:
            # calculator cross entropy on all data set
            total_cross_entropy = sess.run(cross_entropy,
                                           feed_dict={x: X, y_: Y})
            print "After {} training steps, " \
                  "cross entropy on all data is {}:" \
                  "".format(i, total_cross_entropy)

    # show the network parameters after training
    print sess.run(w1)
    print sess.run(w2)



