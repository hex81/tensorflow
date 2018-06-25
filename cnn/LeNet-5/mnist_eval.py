#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 3/30/18 5:16 PM
# @Author  : Xin He
import time
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

import mnist_inference
import mnist_train


# load latest model every 10s, and test on it.
EVAL_INTERVAL_SECS = 10


def evaluate(mnist):
    with tf.Graph().as_default() as g:
        # define input and output format
        x = tf.placeholder(tf.float32, [
                                mnist.validation.num_examples,
                                mnist_inference.IMAGE_SIZE,
                                mnist_inference.IMAGE_SIZE,
                                mnist_inference.NUM_CHANNELS],
                           name='x-input')
        y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE],
                            name='y-input')
        reshaped_xs = np.reshape(mnist.validation.images,
                                 (mnist.validation.num_examples,
                                  mnist_inference.IMAGE_SIZE,
                                  mnist_inference.IMAGE_SIZE,
                                  mnist_inference.NUM_CHANNELS))
        validate_feed = {x: reshaped_xs,
                         y_: mnist.validation.labels}

        # calculator forward propagation result without regularization loss.
        # when we test the model, we needn't regularization value.
        y = mnist_inference.inference(x, False, None)

        # calculator correct rate. use tf.argmax(y, 1) to get classification.
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # load model by renaming variables
        variable_averages = tf.train.ExponentialMovingAverage(
            mnist_train.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        # get accuracy every EVAL_INTERVAL_SECS
        while True:
            with tf.Session() as sess:
                # tf.train.get_checkpoint_state find the latest file
                ckpt = tf.train.get_checkpoint_state(
                    mnist_train.MODEL_SAVE_PATH
                )
                if ckpt and ckpt.model_checkpoint_path:
                    # load model
                    saver.restore(
                        sess,
                        ckpt.model_checkpoint_path
                    )
                    # get round number by files
                    global_step = ckpt.model_checkpoint_path\
                                      .split("/")[-1].split("-")[-1]
                    accuracy_score = sess.run(accuracy,
                                              feed_dict=validate_feed)
                    print("After %s training step(s), validation "
                          "accuracy = %g" % (global_step, accuracy_score))
                else:
                    print 'No checkpoint file found.'
                    return

            time.sleep(EVAL_INTERVAL_SECS)


def main(argv=None):
    mnist = input_data.read_data_sets("/home/hexin/tfproject/mnist_data",
                                      one_hot=True)
    evaluate(mnist)


if __name__ == '__main__':
    tf.app.run()
