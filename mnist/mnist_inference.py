#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 3/26/18 4:40 PM
# @Author  : Xin He

import tensorflow as tf


# define neural network parameters
INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500


# get weight variables
def get_weight_variable(shape, regularizer):
    weights = tf.get_variable(
        "weights", shape,
        initializer=tf.truncated_normal_initializer(stddev=0.1))

    # add regularization to losses
    if regularizer is not None:
        tf.add_to_collection('losses', regularizer(weights))

    return weights


# define forward propagation
def inference(input_tensor, regularizer):
    # declare the first layer variables and finish forward propagation
    with tf.variable_scope('layer1'):
        weights = get_weight_variable(
            [INPUT_NODE, LAYER1_NODE], regularizer)
        biases = tf.get_variable("biases", [LAYER1_NODE],
                                 initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

    with tf.variable_scope('layer2'):
        weights = get_weight_variable(
            [LAYER1_NODE, OUTPUT_NODE], regularizer
        )

        biases = tf.get_variable(
            "biases", [OUTPUT_NODE], initializer=tf.constant_initializer(0.0)
        )

        layer2 = tf.matmul(layer1, weights) + biases

    return layer2
