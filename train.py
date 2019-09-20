import os
import time
import sys

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from mil import MIL
from utils import np_sigmoid
from utils import print_metrics
from utils import get_mnist_data
from utils import instances_to_bags

#os.environ["CUDA_VISIBLE_DEVICES"] = "3"
flags = tf.app.flags
FLAGS = flags.FLAGS

def train(opt_dict):
    train_set, valid_set, test_set = get_mnist_data()
    x_train, y_train = instances_to_bags(ds=train_set, 
                                         n_inst=FLAGS.n_inst, 
                                         target=FLAGS.target,
                                         n_bags=FLAGS.n_bags,
                                         p=FLAGS.prob_target)
    x_test, y_test = instances_to_bags(ds=test_set, 
                                       n_inst=FLAGS.n_inst, 
                                       target=FLAGS.target,
                                       n_bags=1000,
                                       p=FLAGS.prob_target)

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
            print_metrics(true_train, pred_train)

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
    flags.DEFINE_integer('n_inst', 20, 'number of instances per bag')
    flags.DEFINE_integer('n_bags', 100, 'number of instances per bag')
    flags.DEFINE_integer('target', 9, 'target integer')
    flags.DEFINE_float('prob_target', 0.5, 'Probability of containing target image')
    flags.DEFINE_bool('use_attn', True, 'Whether to use the attn mechanism')
    flags.DEFINE_bool('use_gate', False, 'Whether to use the gate mechanism')
    flags.DEFINE_float('wd', 1e-4, 'weight_decay')
    flags.DEFINE_float('lr', 5e-4, 'Initial learning rate')
    flags.DEFINE_integer('n_epoches', 100, 'Epoch size')
    flags.DEFINE_bool('save_model', False, '')
    main()
