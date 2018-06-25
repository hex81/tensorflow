#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 3/29/18 11:32 PM
# @Author  : Xin He

import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import mnist_inference
import numpy as np


# neural network parameter configuration
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.01     # learning rate should be small.
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99

MODEL_SAVE_PATH = "/home/hexin/PycharmProjects/tensorflow/cnn/LeNet-5"
MODEL_NAME = "model.ckpt"


def train(mnist):
    # define input & output placeholder
    x = tf.placeholder(tf.float32, [
            BATCH_SIZE,
            mnist_inference.IMAGE_SIZE,
            mnist_inference.IMAGE_SIZE,
            mnist_inference.NUM_CHANNELS],
        name='x-input')
    y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE],
                        name='y-input')

    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)

    # use forward propagation which defined in mnist_inference.py
    y = mnist_inference.inference(x, True, regularizer)
    global_step = tf.Variable(0, trainable=False)

    # define cross entropy, learning rate, moving average and training progress
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,
                                                          global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE, global_step, mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY
    )

    train_step = tf.train.GradientDescentOptimizer(
        learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')

    # initial persistence class
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        # training model
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            reshaped_xs = np.reshape(xs, (BATCH_SIZE,
                                          mnist_inference.IMAGE_SIZE,
                                          mnist_inference.IMAGE_SIZE,
                                          mnist_inference.NUM_CHANNELS))
            _, loss_value, step = sess.run([train_op, loss, global_step],
                                           feed_dict={x: reshaped_xs, y_: ys})

            # save model every 1000 times
            if i % 1000 == 0:
                # print training result
                print("After %d training step(s), loss on training "
                      "batch is %g." % (step, loss_value))
                saver.save(
                    sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME),
                    global_step=global_step)


def main(argv=None):
    mnist = input_data.read_data_sets("/home/hexin/tfproject/mnist_data",
                                      one_hot=True)
    train(mnist)


if __name__ == '__main__':
    tf.app.run()

