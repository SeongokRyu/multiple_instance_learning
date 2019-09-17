from easydict import EasyDict
import tensorflow as tf

from layers import conv2d
from layers import dense
from layers import max_pool

class MIL:
    def __init__(self, hw, ch, attn_dim, use_attn, use_gate, wd, lr):
        self.hw = hw
        self.ch = ch
        self.attn_dim = attn_dim
        self.use_attn = use_attn
        self.use_gate = use_gate
        self.wd = wd
        self.lr = lr

    def encoder(self, x):
        out = conv2d(x, 20, 5, activation=tf.nn.relu)
        out = max_pool(out, 2, 2)
        out = conv2d(out, 50, 5, activation=tf.nn.relu)
        out = max_pool(out, 2, 2)
        out = tf.layers.flatten(out)
        out = dense(out, 500, activation=tf.nn.relu)
        return out

    def attention(self, x):
        x_shape = tf.shape(x)
        w1 = tf.get_variable(name='attn_w1', 
                             shape=[500, self.attn_dim],
                             initializer=tf.glorot_normal_initializer())
        w2 = tf.get_variable(name='attn_w2', 
                             shape=[self.attn_dim, 1],
                             initializer=tf.glorot_normal_initializer())

        x = tf.expand_dims(x, axis=0) # out_dim : [1, n_imgs, f]
        out = tf.matmul(x, w1) # out_dim : [1, n_imgs, attn_dim]

        if self.use_gate:
            w3 = tf.get_variable(name='attn_w3', 
                                 shape=[500, self.attn_dim],
                                 initializer=tf.glorot_normal_initializer())
            gate = tf.matmul(x, w3)
            out = tf.multiply(out, gate)

        out = tf.matmul(out, w2) # out_dim : [1, n_imgs, 1]
        out = tf.layers.flatten(out)
        attn = tf.nn.softmax(out, axis=1)
        coef = tf.expand_dims(attn, axis=2)
        coef = tf.concat([coef for i in range(500)], axis=2)
        embedding = tf.multiply(coef, x)
        embedding = tf.reduce_mean(embedding, axis=1)
        return  embedding, attn

    def get_opt_dict(self):
        x = tf.placeholder(tf.float32, shape=[None, self.hw, self.hw])
        y = tf.placeholder(tf.float32, shape=[1, ])
        is_training = tf.placeholder(tf.bool, shape=[])

        embedding = self.encoder(tf.expand_dims(x,-1))
        attn = None
        if self.use_attn:
            embedding, attn = self.attention(embedding)
        
        else:
            embedding = tf.reduce_mean(
                tf.expand_dims(embedding, axis=0), axis=1)

        logits = dense(embedding, 1)
        logits = tf.reshape(logits, [-1])

        loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits))

        var_total = tf.trainable_variables()
        decay_var = [v for v in var_total if 'kernel' in v.name]
        optimizer = tf.contrib.opt.AdamWOptimizer(
            weight_decay=self.wd, learning_rate=self.lr)
        train_op = optimizer.minimize(loss=loss, var_list=var_total, 
                                      decay_var_list=decay_var)

        return EasyDict(
            x=x, y=y, is_training=is_training, attn=attn,
            logits=logits, loss=loss, train_op=train_op)    
        



