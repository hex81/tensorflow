#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 4/24/18 4:44 PM
# @Author  : Xin He
import tensorflow as tf

# define NN parameters
INPUT_NODE = 784    # 28*28
OUTPUT_NODE = 10    # 0-9

IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10

# The first convolution layer size & depth
CONV1_DEEP = 32
CONV1_SIZE = 5

# The second convolution layer size & depth
CONV2_DEEP = 64
CONV2_SIZE = 5

# Full connection layer size
FC_SIZE = 512


# Define CNN forward propagation. Use parameter train to distinguish training
# and testing progress. Use dropout method to prevent over fit.
def inference(input_tensor, train, regularizer):
    # Define the first convolution layer and forward propagation.
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable(
            "weight", [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        conv1_biases = tf.get_variable(
            "bias", [CONV1_DEEP], initializer=tf.constant_initializer(0.0)
        )
        conv1 = tf.nn.conv2d(
            input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME'
        )
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    # Define the second layer -- pooling layer, and forward propagation.
    with tf.name_scope('layer2-pool1'):
        pool1 = tf.nn.max_pool(
            relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Define the third layer -- convolution layer and forward propagation.
    with tf.variable_scope('layer3-conv2'):
        conv2_weights = tf.get_variable(
            "weight", [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        conv2_biases = tf.get_variable(
            "bias", [CONV2_DEEP], initializer=tf.constant_initializer(0.0)
        )
        conv2 = tf.nn.conv2d(
            pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME'
        )
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    # Define the fourth layer -- pooling layer, and forward propagation.
    # input is 14*14*64 array, output is 7*7*64 array.
    with tf.name_scope('layer4-pool2'):
        pool2 = tf.nn.max_pool(
            relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # transform output to input format for fifth layer.
    pool_shape = pool2.get_shape().as_list()

    # calc vector length, pool_shape[0] is the number of one batch
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]

    # reshape data
    reshaped = tf.reshape(pool2, [pool_shape[0], nodes])

    # define full connection layer & propagation
    with tf.variable_scope('layer5-fc1'):
        fc1_weights = tf.get_variable(
            "weight", [nodes, FC_SIZE],
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        if regularizer is not None:
            tf.add_to_collection('losses', regularizer(fc1_weights))
        fc1_biases = tf.get_variable(
            "bias", [FC_SIZE], initializer=tf.constant_initializer(0.1)
        )

        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        if train:
            fc1 = tf.nn.dropout(fc1, 0.5)

    # define sixth full connection layer & forward propagation.
    with tf.variable_scope('layer6-fc2'):
        fc2_weights = tf.get_variable(
            "weight", [FC_SIZE, NUM_LABELS],
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        if regularizer is not None:
            tf.add_to_collection('losses', regularizer(fc2_weights))

        fc2_biases = tf.get_variable(
            "bias", [NUM_LABELS],
            initializer=tf.constant_initializer(0.1)
        )

        logit = tf.matmul(fc1, fc2_weights) + fc2_biases

    return logit








