#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 5/29/18 10:42 AM
# @Author  : Xin He


import glob
import os.path
import numpy as np
import tensorflow as tf


# raw data path
INPUT_DATA = '/home/hexin/PycharmProjects/tensorflow/' \
             'cnn/TransferLearning/flower_photos'

# output path
OUTPUT_FILE = '/home/hexin/PycharmProjects/tensorflow/' \
              'cnn/TransferLearning/flower_processed_data.npy'

# ratio of test data & validation data
VALIDATION_PERCENTAGE = 10
TEST_PERCENTAGE = 10


# divided data into training data, validation data & test data.
def create_image_lists(sess, testing_percentage, validation_percentage):
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]
    is_root_dir = True

    # various data set
    training_images = []
    training_labels = []
    testing_images = []
    testing_labels = []
    validation_images = []
    validation_labels = []
    current_label = 0

    # read all sub dirs
    file_num = 0
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue

        # get all pictures from one subdir
        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
        file_list = []
        dir_name = os.path.basename(sub_dir)
        # current_label = dir_name.lower()
        for extension in extensions:
            file_glob = os.path.join(INPUT_DATA, dir_name, '*.' + extension)
            file_list.extend(glob.glob(file_glob))
            if file_list is None:
                continue

        print "{} file num is: {}".format(dir_name, len(file_list))
        # process pictures
        for file_name in file_list:
            # print "{} file is: {}".format(file_num, file_name)
            # transfer picture to 299*299, so that inception-v3 can process
            # it.
            # image_raw_data = gfile.FastGFile(file_name, 'rb').read()
            # image = tf.image.decode_jpeg(image_raw_data)
            # if image.dtype != tf.float32:
            #     image = tf.image.convert_image_dtype(image,
            #                                          dtype=tf.float32)
            # image = tf.image.resize_images(image, [299, 299])
            # image_value = sess.run(image)

            # divide data random
            chance = np.random.randint(100)
            if chance < validation_percentage:
                validation_images.append(file_name)
                validation_labels.append(current_label)
            elif chance < (testing_percentage + validation_percentage):
                testing_images.append(file_name)
                testing_labels.append(current_label)
            else:
                training_images.append(file_name)
                training_labels.append(current_label)

            file_num += 1

        current_label += 1

    # shuffle training data
    state = np.random.get_state()
    np.random.shuffle(training_images)
    np.random.set_state(state)
    np.random.shuffle(training_labels)

    print "training file number is: {}".format(len(training_images))
    print "validation file number is: {}".format(len(validation_images))
    print "testing file number is: {}".format(len(testing_images))
    return np.asarray([training_images, training_labels,
                       validation_images, validation_labels,
                       testing_images, testing_labels])


def main():
    with tf.Session() as sess:
        processed_data = create_image_lists(sess, TEST_PERCENTAGE,
                                            VALIDATION_PERCENTAGE)
        np.save(OUTPUT_FILE, processed_data)


if __name__ == '__main__':
    main()

