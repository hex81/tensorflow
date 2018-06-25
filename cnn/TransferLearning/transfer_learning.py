#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 5/30/18 5:29 PM
# @Author  : Xin He

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
import tensorflow.contrib.slim as slim


# load inception_v3 model
import tensorflow.contrib.slim.python.slim.nets.inception_v3 as inception_v3


# data file
INPUT_DATA = '/home/hexin/PycharmProjects/tensorflow/cnn/TransferLearning/' \
             'flower_processed_data.npy'
TRAIN_FILE = '/home/hexin/PycharmProjects/tensorflow/cnn/TransferLearning/' \
             'save_model'
# google model file
CKPT_FILE = '/home/hexin/PycharmProjects/tensorflow/cnn/TransferLearning/' \
            'inception_v3.ckpt'

# defile training parameter
LEARNING_RATE = 0.0001
STEPS = 200
BATCH = 10
N_CLASSES = 5

# exclude trained parameter in google model
CHECKPOINT_EXCLUDE_SCOPES = 'InceptionV3/Logits,InceptionV3/AuxLogits'
# trainable parameter
TRAINABLE_SCOPES = 'InceptionV3/Logits,InceptionV3/AuxLogits'


# get all parameters from google model
def get_tuned_variables():
    exclusions = [scope.strip() for scope in
                  CHECKPOINT_EXCLUDE_SCOPES.split(',')]
    variables_to_restore = []
    # remove excluded parameters
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)
    return variables_to_restore


# get all trainable parameters
def get_trainable_variables():
    scopes = [scope.strip() for scope in TRAINABLE_SCOPES.split(',')]
    variables_to_train = []
    for scope in scopes:
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        variables_to_train.extend(variables)

    return variables_to_train


def main():
    # load data
    processed_data = np.load(INPUT_DATA)
    training_images = processed_data[0]
    n_training_example = len(training_images)
    training_labels = processed_data[1]
    validation_images = processed_data[2]
    validatioan_labels = processed_data[3]
    testing_images = processed_data[4]
    testing_labels = processed_data[5]

    print "%d training examples, %d validation examples and %d examples." % \
          (n_training_example, len(validatioan_labels), len(testing_labels))

    # define inception-v3 input
    images = tf.placeholder(tf.float32, [None, 299, 299, 3],
                            name='input_images')
    labels = tf.placeholder(tf.int64, [None], name='labels')

    # define inception-v3 model
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits, _ = inception_v3.inception_v3(images, num_classes=N_CLASSES)

    # get trainable variables
    trainable_variables = get_trainable_variables()

    # define cross entropy.
    tf.losses.softmax_cross_entropy(tf.one_hot(labels, N_CLASSES),
                                    logits, weights=1.0)

    # define training process.
    train_step = tf.train.RMSPropOptimizer(
        LEARNING_RATE).minimize(tf.losses.get_total_loss())

    # calc accuracy
    with tf.name_scope('evaluation'):
        correct_prediction = tf.equal(tf.argmax(logits, 1), labels)
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction,
                                                 tf.float32))

    # define model load function
    load_fn = slim.assign_from_checkpoint_fn(CKPT_FILE,
                                             get_tuned_variables(),
                                             ignore_missing_vars=True)

    # define function to save trained model
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # initial variables
        init = tf.global_variables_initializer()
        sess.run(init)

        # load trained google model
        print 'loading tuned variables from {}'.format(CKPT_FILE)
        load_fn(sess)

        start = 0
        end = BATCH

        validation_images_value = []
        print "Validation set num is: {}.".format(len(validation_images))
        for file_name in validation_images:
            # print "Get validation set {} value.".format(file_name)
            image_raw_data = gfile.FastGFile(file_name, 'rb').read()
            image = tf.image.decode_jpeg(image_raw_data)
            if image.dtype != tf.float32:
                image = tf.image.convert_image_dtype(image,
                                                     dtype=tf.float32)
            image = tf.image.resize_images(image, [299, 299])
            image_value = sess.run(image)
            validation_images_value.append(image_value)

        for i in xrange(STEPS):
            # transfer picture to 299*299, so that inception-v3 can process
            # it.
            training_images_value = []
            for file_name in training_images[start:end]:
                # print "Get training set {} value.".format(file_name)
                image_raw_data = gfile.FastGFile(file_name, 'rb').read()
                image = tf.image.decode_jpeg(image_raw_data)
                if image.dtype != tf.float32:
                    image = tf.image.convert_image_dtype(image,
                                                         dtype=tf.float32)
                image = tf.image.resize_images(image, [299, 299])
                image_value = sess.run(image)
                training_images_value.append(image_value)

            sess.run(train_step, feed_dict={
                images: training_images_value,
                labels: training_labels[start:end]
            })

            print "Training {} steps finished.".format(i)

            if i % 50 == 0 or i + 1 == STEPS:
                saver.save(sess, TRAIN_FILE, global_step=i)
                validation_accuracy = sess.run(
                    evaluation_step, feed_dict={images: validation_images_value,
                                                labels: validatioan_labels
                                                })
                print "Step %d: validation accuracy = %.1f%%" % \
                      (i, validation_accuracy * 100.0)

            start = end
            if start == n_training_example:
                start = 0
            end = start + BATCH

            if end > n_training_example:
                end = n_training_example

        testing_images_value = []
        for file_name in testing_images:
            # print "Get testing set {} value.".format(file_name)
            image_raw_data = gfile.FastGFile(file_name, 'rb').read()
            image = tf.image.decode_jpeg(image_raw_data)
            if image.dtype != tf.float32:
                image = tf.image.convert_image_dtype(image,
                                                     dtype=tf.float32)
            image = tf.image.resize_images(image, [299, 299])
            image_value = sess.run(image)
            testing_images_value.append(image_value)

        test_accuracy = sess.run(evaluation_step, feed_dict={
            images: testing_images_value, labels: testing_labels
        })

        print "Final test accuracy = %.1f%%" % (test_accuracy * 100)


if __name__ == '__main__':
    main()
