#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 3/19/18 11:22 PM
# @Author  : Xin He

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# constant of MNIST data
INPUT_NODE = 784  # The node on input layer. 28 * 28
OUTPUT_NODE = 10  # The node on output layer. 0 ~ 9

# Neural network layer parameters
LAYER1_NODE = 500  # The node on hidden layer.
BATCH_SIZE = 100  # Training number.
LEARNING_RATE_BASE = 0.8  # The basic learning rate.
LEARNING_RATE_DECAY = 0.99  # The decay of learning rate.
REGULARIZATION_RATE = 0.0001  # The coefficient of regularization
TRAINING_STEPS = 30000  # The round of training
MOVING_AVERAGE_DECAY = 0.99  # The rate of moving average.


# define a function to calculator forward propagation result.
# Use ReLU to complete delinearization.
def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    # If there is no moving average class, we'll use parameter original value.
    if avg_class is None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        # return forward propagation result.
        return tf.matmul(layer1, weights2) + biases2

    else:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1))
                            + avg_class.average(biases1))
        return tf.matmul(layer1,
                         avg_class.average(weights2)
                         ) + avg_class.average(biases2)


# train model
def train(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')

    # generate hidden layer parameters.
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE],
                                               stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))

    # generate output layer parameters
    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE],
                                               stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    # calculator forward propagation result without moving average
    y = inference(x, None, weights1, biases1, weights2, biases2)

    # define training steps
    global_step = tf.Variable(0, trainable=False)

    # initialize moving average class
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,
                                                          global_step)

    # apply moving average on all trainable parameters
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    # calculator forward propagation result with moving average
    average_y = inference(x, variable_averages,
                          weights1, biases1, weights2, biases2)

    # calculator cross entropy
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=y, labels=tf.argmax(y_, 1))

    # calculator cross entropy mean of batch data
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # calculator L2 regularization loss function
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

    # calculator regularization loss
    regularization = regularizer(weights1) + regularizer(weights2)

    # calculator total loss
    loss = cross_entropy_mean + regularization

    # set exponent decay learning rate
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE, global_step,
        mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY)

    # optimize loss
    train_step = tf.train.GradientDescentOptimizer(
        learning_rate).minimize(loss, global_step=global_step)

    train_op = tf.group(train_step, variable_averages_op)

    # compare prediction value and input value
    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))

    # calculator correct mean
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # initial session and start training progress
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        # prepare validation data
        validate_feed = {x: mnist.validation.images,
                         y_: mnist.validation.labels}

        # prepare test data
        test_feed = {x: mnist.test.images, y_: mnist.test.labels}

        # train neural network
        for i in xrange(TRAINING_STEPS):
            # output test result on validation data every 1000 round
            if i % 1000 == 0:
                # calculator moving average on validation data.
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print("After %d training step(s), validation accuracy " 
                      "using average model is %g " % (i, validate_acc))

            # generate batch training data, run training progress
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xs, y_: ys})

        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print("After %d training step(s), test accuracy using average "
              "model is %g" % (TRAINING_STEPS, test_acc))


# main function
def main(argv=None):
    # load mnist data
    mnist = input_data.read_data_sets("/home/hexin/tfproject/mnist_data",
                                      one_hot=True)
    train(mnist)


# tf.app.run() will call main()
if __name__ == '__main__':
    tf.app.run()
