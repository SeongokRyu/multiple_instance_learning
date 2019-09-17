import tensorflow as tf

def conv2d(x, filters, kernel_size, strides=1, padding='same', activation=None):
    conv_args = dict(filters=filters, 
                     kernel_size=kernel_size, 
                     strides=strides, 
                     padding=padding, 
                     activation=activation,
                     kernel_initializer=tf.glorot_normal_initializer())
    return tf.layers.conv2d(x, **conv_args)

def dense(x, dim, use_bias=True, activation=None):
    dense_args = dict(units=dim,
                      use_bias=use_bias,
                      activation=activation,
                      kernel_initializer=tf.glorot_normal_initializer())
    return tf.layers.dense(x, **dense_args)

def max_pool(x, pool_size, strides, padding='same'):
    pool_args = dict(pool_size=pool_size,
                     strides=strides,
                     padding=padding)
    return tf.layers.max_pooling2d(x, **pool_args)    
