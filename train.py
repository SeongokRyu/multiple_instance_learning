import os
import time
import sys

import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from mil import MIL
from tensorflow.examples.tutorials.mnist import input_data
from utils import np_sigmoid
from utils import print_metrics

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
flags = tf.app.flags
FLAGS = flags.FLAGS
def get_data():
    path = "/home/wykgroup/seongok/ml_study/mnist_mlp/mnist/"
    mnist = input_data.read_data_sets(path, one_hot=False)
    train_set = mnist.train
    valid_set = mnist.validation
    test_set = mnist.test
    return train_set, valid_set, test_set

def instances_to_bag(ds, n_inst, target, n_bags=None):
    x = np.asarray([img.reshape(28,28) for img in ds.images])
    y = ds.labels

    n_total = x.shape[0]
    indices = np.random.randint(n_total, size=n_total)
    if n_bags is None:
        n_bags = n_total//n_inst

    x = x[indices]
    y = y[indices]

    x_bag = []
    y_bag = []
    for i in range(n_bags):
        xi = x[i*n_inst:(i+1)*n_inst]
        yi = y[i*n_inst:(i+1)*n_inst]

        label = 0.0
        if target in yi:
            label = 1.0
        x_bag.append(xi)
        y_bag.append(label)
    return np.asarray(x_bag), np.asarray(y_bag)

def train(opt_dict):
    train_set, valid_set, test_set = get_data()
    x_train, y_train = instances_to_bag(
        train_set, FLAGS.n_inst, FLAGS.target, n_bags=FLAGS.n_train_bags)
    x_valid, y_valid = instances_to_bag(valid_set, FLAGS.n_inst, FLAGS.target)
    x_test, y_test = instances_to_bag(valid_set, FLAGS.n_inst, FLAGS.target)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(FLAGS.n_epoches):
            # Train
            true_train = np.empty([0,])
            pred_train = np.empty([0,])
            for i in range(x_train.shape[0]):
                st_i = time.time()
                xi = x_train[i]
                yi = np.asarray([y_train[i]])

                feed_dict = {opt_dict.x:xi, opt_dict.y:yi}
                ops = [opt_dict.train_op, opt_dict.loss, opt_dict.logits] 
                _, loss, logits = sess.run(ops, feed_dict=feed_dict)
                et_i = time.time()
                print ("Training", epoch, "-th epoch\t", \
                       i, "-th bag\t Loss=", loss, 
                       "\t Time:", round(et_i-st_i,3), "(s)")

                true_train = np.concatenate([true_train, yi], axis=0)
                pred_train = np.concatenate([pred_train, np_sigmoid(logits)], axis=0)

            # Validation
            true_valid = np.empty([0,])
            pred_valid = np.empty([0,])
            for i in range(x_train.shape[0]):
                st_i = time.time()
                xi = x_valid[i]
                yi = np.asarray([y_valid[i]])

                feed_dict = {opt_dict.x:xi, opt_dict.y:yi}
                ops = [opt_dict.loss, opt_dict.logits]
                loss, logits = sess.run(ops, feed_dict=feed_dict)
                et_i = time.time()
                print ("Validation", epoch, "-th epoch\t", \
                       i, "-th bag\t Loss=", loss, 
                       "\t Time:", round(et_i-st_i,3), "(s)")

                true_valid = np.concatenate([true_valid, yi], axis=0)
                pred_valid = np.concatenate([pred_valid, np_sigmoid(logits)], axis=0)
            print_metrics(true_train, pred_train)
            print_metrics(true_valid, pred_valid)

        # Test
        true_test = np.empty([0,])
        pred_test = np.empty([0,])
        for i in range(x_train.shape[0]):
            st_i = time.time()
            xi = x_test[i]
            yi = np.asarray([y_test[i]])

            feed_dict = {opt_dict.x:xi, opt_dict.y:yi}
            ops = [opt_dict.loss, opt_dict.logits]
            loss, logits = sess.run(ops, feed_dict=feed_dict)
            et_i = time.time()
            print ("Validation", epoch, "-th epoch\t", \
                   i, "-th bag\t Loss=", loss, 
                   "\t Time:", round(et_i-st_i,3), "(s)")

            true_test = np.concatenate([true_test, yi], axis=0)
            pred_test = np.concatenate([pred_test, np_sigmoid(logits)], axis=0)
        print_metrics(true_test, pred_test)
        print ("Finish training and test")
    return

def main():
    model = MIL(hw=FLAGS.hw, 
                ch=FLAGS.ch,
                attn_dim=FLAGS.attn_dim, 
                use_attn=FLAGS.use_attn, 
                use_gate=FLAGS.use_gate, 
                wd=FLAGS.wd, 
                lr=FLAGS.lr)
    opt_dict = model.get_opt_dict()
    train(opt_dict)
    return

if __name__ == '__main__':
    flags.DEFINE_integer('hw', 28, 'height and width')
    flags.DEFINE_integer('ch', 1, 'channel')
    flags.DEFINE_integer('attn_dim', 128, 'hidden dimension of attention')
    flags.DEFINE_integer('n_inst', 50, 'number of instances per bag')
    flags.DEFINE_integer('n_train_bags', 100, 'number of instances per bag')
    flags.DEFINE_integer('target', 9, 'target integer')
    flags.DEFINE_bool('use_attn', True, 'Whether to use the attn mechanism')
    flags.DEFINE_bool('use_gate', False, 'Whether to use the gate mechanism')
    flags.DEFINE_float('wd', 1e-4, 'weight_decay')
    flags.DEFINE_float('lr', 5e-4, 'Initial learning rate')
    flags.DEFINE_integer('n_epoches', 100, 'Epoch size')
    flags.DEFINE_bool('save_model', False, '')
    main()
