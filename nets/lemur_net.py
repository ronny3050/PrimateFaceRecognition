from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf
import tensorflow.contrib.slim as slim

model_params = {
    '4': ([0, 0, 0, 0], [64, 128, 256, 512], [1,1,1,1]),
    '10': ([0, 1, 2, 0], [64, 128, 256, 512]),
    '20': ([1, 2, 4, 1], [64, 128, 256, 512]),
    '36': ([2, 4, 8, 2], [64, 128, 256, 512]),
    '64': ([3, 8, 16, 3], [64, 128, 256, 512]),
    'lemur': ([0, 0, 0, 0], [64, 256, 512, 1024], [1,32,32,32,32]),
    'lemurDropout': ([0,0,0,0], [64,128,256,512], [1,1,1,1,1])
}

batch_norm_params = {
    # Decay for the moving averages.
    'decay': 0.995,
    # epsilon to prevent 0s in variance.
    'epsilon': 0.001,
    # force in-place updates of mean and variance estimates
    'updates_collections': None,
    # Moving averages ends up in the trainable variables collection
    'variables_collections': [ tf.GraphKeys.TRAINABLE_VARIABLES ],
}   

batch_norm_params_last = {
    # Decay for the moving averages.
    'decay': 0.995,
    # epsilon to prevent 0s in variance.
    'epsilon': 10e-8,
    # force in-place updates of mean and variance estimates
    'center': False,
    # not use beta
    'scale': False,
    # not use gamma
    'updates_collections': None,
    # Moving averages ends up in the trainable variables collection
 
   'variables_collections': [ tf.GraphKeys.TRAINABLE_VARIABLES ],
}


trans_conv_args = {
    'weights_initializer': slim.xavier_initializer(),
    'biases_initializer': tf.constant_initializer(0.0)
}

res_conv_args = {
    'weights_initializer': tf.truncated_normal_initializer(stddev=0.01),
    'biases_initializer': None
}

fc_args = {
    'weights_initializer': slim.xavier_initializer(),
    'biases_initializer': tf.constant_initializer(0.0),
    'activation_fn': None,
    'normalizer_fn': None,
}


def convolution(net, num_kernels, kernel_size, groups=1, shuffle=False, 
        stride=1, padding='SAME', scope=None, xargs=trans_conv_args):
    assert num_kernels % groups == 0, '%d %d' % (num_kernels, groups)
    if groups==1:
        net = slim.conv2d(net, num_kernels, kernel_size=kernel_size, stride=stride, padding=padding, scope=scope, **xargs)
        return slim.dropout(net)
    else:
        with tf.variable_scope(scope, 'group_conv'):
            num_kernels_split = int(num_kernels / groups)
            input_splits = tf.split(net, groups, axis=3)
            output_splits = [slim.conv2d(input_split, num_kernels_split, 
                    kernel_size=kernel_size, stride=stride, padding=padding, **xargs)
                    for input_split in input_splits]
            output = tf.concat(output_splits, axis=3)
            if shuffle:
                output = channel_shuffle('shuffle', output, groups)
            return output

def channel_shuffle(name, x, num_groups):
    with tf.variable_scope(name) as scope:
        n, h, w, c = x.shape.as_list()
        x_reshaped = tf.reshape(x, [-1, h, w, num_groups, c // num_groups])
        x_transposed = tf.transpose(x_reshaped, [0, 1, 2, 4, 3])
        output = tf.reshape(x_transposed, [-1, h, w, c])
        return output

def parametric_relu(x):
    num_channels = x.shape[-1].value
    with tf.variable_scope('PRELU'):
        alpha = tf.get_variable('alpha', (1,1,1,num_channels),
                        initializer=tf.constant_initializer(0.0),
                        dtype=tf.float32)
        mask = x>=0
        mask_pos = tf.cast(mask, tf.float32)
        mask_neg = tf.cast(tf.logical_not(mask), tf.float32)
        return mask_pos * x + mask_neg * alpha * x

def se_module(input_net, ratio=16, reuse = None, scope = None):
    with tf.variable_scope(scope, 'SE', [input_net], reuse=reuse):
        h,w,c = tuple([dim.value for dim in input_net.shape[1:4]])
        assert c % ratio == 0
        hidden_units = int(c / ratio)
        squeeze = slim.avg_pool2d(input_net, [h,w], padding='VALID')
        excitation = slim.flatten(squeeze)
        excitation = slim.fully_connected(excitation, hidden_units, scope='se_fc1',
                                weights_initializer=slim.xavier_initializer(), 
                                activation_fn=tf.nn.relu)
        excitation = slim.fully_connected(excitation, c, scope='se_fc2',
                                weights_initializer=slim.xavier_initializer(), 
                                activation_fn=tf.nn.sigmoid)        
        excitation = tf.reshape(excitation, [-1,1,1,c])
        output_net = input_net * excitation

        return output_net


# activation = parametric_relu
activation = lambda x: tf.keras.layers.PReLU(shared_axes=[1,2]).apply(x)
# activation = tf.nn.softplus
# activation = tf.nn.relu

def conv_module(net, num_res_layers, num_kernels, groups, reuse = None, scope = None):
    with tf.variable_scope(scope, 'conv', [net], reuse=reuse):
        net = convolution(net, num_kernels, kernel_size=3, groups=groups, shuffle=False,
                        stride=1, padding='SAME', scope='transform', xargs=trans_conv_args)
        net = slim.max_pool2d(net, 3, stride=2, padding='SAME')
        shortcut = net
        for i in range(num_res_layers):
            # num_kernels_sm = int(num_kernels / 2)
            net = convolution(net, num_kernels, kernel_size=1, groups=groups, shuffle=True,
                            stride=1, padding='SAME', scope='res_%d_1'%i, xargs=res_conv_args)
            net = convolution(net, num_kernels, kernel_size=3, groups=groups, shuffle=False,
                            stride=1, padding='SAME', scope='res_%d_2'%i, xargs=res_conv_args)
            print('| ---- block_%d' % i)
            net = se_module(net)
            net = net + shortcut
            shortcut = net
    return net

def inference(images, keep_probability, phase_train=True, bottleneck_layer_size=512, 
            weight_decay=0.0, reuse=None, model_version=None):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        activation_fn=activation,
                        normalizer_fn=None,
                        normalizer_params=None):
        with slim.arg_scope([slim.dropout],
                            keep_prob=keep_probability,
                            is_training=phase_train):
            with tf.variable_scope('SphereNet', [images], reuse=reuse):
                with slim.arg_scope([slim.batch_norm, slim.dropout],
                                    is_training=phase_train):
                    print('SphereNet input shape:', [dim.value for dim in images.shape])
                    
                    model_version = '4' if model_version ==None else model_version
                    num_layers, num_kernels, groups = model_params[model_version]

                    net = conv_module(images, num_layers[0], num_kernels[0], groups[0], scope='conv1')
                    print('module_1 shape:', [dim.value for dim in net.shape])

                    net = conv_module(net, num_layers[1], num_kernels[1], groups[1], scope='conv2')
                    print('module_2 shape:', [dim.value for dim in net.shape])
                    
                    net = conv_module(net, num_layers[2], num_kernels[2], groups[2], scope='conv3')
                    print('module_3 shape:', [dim.value for dim in net.shape])

                    net = conv_module(net, num_layers[3], num_kernels[3], groups[3], scope='conv4')
                    print('module_4 shape:', [dim.value for dim in net.shape])
                    
                    # net = slim.avg_pool2d(net, 7)
                    net = convolution(net, bottleneck_layer_size, kernel_size=[net.shape[1], net.shape[2]], groups=groups[4], shuffle=False,
                                    stride=1, padding='VALID', scope='bottleneck', xargs=fc_args)
                    
                    #net = slim.dropout(net, keep_probability, is_training = phase_train)
                    net = slim.flatten(net)

                    # net= slim.batch_norm(net, **batch_norm_params_last)

                    with tf.device(None):
                        tf.summary.histogram('unormed_prelogits', net)

    return net
