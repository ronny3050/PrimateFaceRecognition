"""Main training file for face recognition
"""
# MIT License
# 
# Copyright (c) 2018 Debayan Deb
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

import utils
import tflib

class Network:
    def __init__(self):
        self.graph = tf.Graph()
        gpu_options = tf.GPUOptions(allow_growth=True)
        tf_config = tf.ConfigProto(gpu_options=gpu_options,
                allow_soft_placement=True, log_device_placement=False)
        self.sess = tf.Session(graph=self.graph, config=tf_config)
            
    def initialize(self, config, num_classes):
        '''
            Initialize the graph from scratch according config.
        '''
        with self.graph.as_default():
            with self.sess.as_default():
                # Set up placeholders
                w, h = config.image_size
                channels = config.channels
                image_batch_placeholder = tf.placeholder(tf.float32, shape=[None, h, w, channels], name='image_batch')
                label_batch_placeholder = tf.placeholder(tf.int32, shape=[None], name='label_batch')
                learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')
                keep_prob_placeholder = tf.placeholder(tf.float32, name='keep_prob')
                phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
                global_step = tf.Variable(0, trainable=False, dtype=tf.int32, name='global_step')

                image_splits = tf.split(image_batch_placeholder, config.num_gpus)
                label_splits = tf.split(label_batch_placeholder, config.num_gpus)
                grads_splits = []
                split_dict = {}
                def insert_dict(k,v):
                    if k in split_dict: split_dict[k].append(v)
                    else: split_dict[k] = [v]
                        
                for i in range(config.num_gpus):
                    scope_name = '' if i==0 else 'gpu_%d' % i
                    with tf.name_scope(scope_name):
                        with tf.variable_scope('', reuse=i>0):
                            with tf.device('/gpu:%d' % i):
                                images = tf.identity(image_splits[i], name='inputs')
                                labels = tf.identity(label_splits[i], name='labels')
                                # Save the first channel for testing
                                if i == 0:
                                    self.inputs = images
                                
                                # Build networks
                                if config.localization_net is not None:
                                    localization_net = utils.import_file(config.localization_net, 'network')
                                    imsize = (112, 112)
                                    images, theta = localization_net.inference(images, imsize, 
                                                    phase_train_placeholder,
                                                    weight_decay = 0.0)
                                    images = tf.identity(images, name='transformed_image')
                                    if i == 0:
                                        tf.summary.image('transformed_image', images)
                                else:
                                    images = images

                                network = utils.import_file(config.network, 'network')
                                prelogits = network.inference(images, keep_prob_placeholder, phase_train_placeholder,
                                                        bottleneck_layer_size = config.embedding_size, 
                                                        weight_decay = config.weight_decay, 
                                                        model_version = config.model_version)
                                prelogits = tf.identity(prelogits, name='prelogits')
                                embeddings = tf.nn.l2_normalize(prelogits, dim=1, name='embeddings')
                                if i == 0:
                                    self.outputs = tf.identity(embeddings, name='outputs')

                                # Build all losses
                                losses = []

                                # Orignal Softmax
                                if 'softmax' in config.losses.keys():
                                    logits = slim.fully_connected(prelogits, num_classes, 
                                                                    weights_regularizer=slim.l2_regularizer(config.weight_decay),
                                                                    # weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                                                    weights_initializer=slim.xavier_initializer(),
                                                                    biases_initializer=tf.constant_initializer(0.0),
                                                                    activation_fn=None, scope='Logits')
                                    cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                                                    labels=labels, logits=logits), name='cross_entropy')
                                    losses.append(cross_entropy)
                                    insert_dict('sloss', cross_entropy)
                                # L2-Softmax
                                if 'cosine' in config.losses.keys():
                                    logits, cosine_loss = tflib.cosine_softmax(prelogits, labels, num_classes, 
                                                            gamma=config.losses['cosine']['gamma'], 
                                                            weight_decay=config.weight_decay)
                                    losses.append(cosine_loss)
                                    insert_dict('closs', cosine_loss)
                                # A-Softmax
                                if 'angular' in config.losses.keys():
                                    a_cfg = config.losses['angular']
                                    angular_loss = tflib.angular_softmax(prelogits, labels, num_classes, 
                                                            global_step, a_cfg['m'], a_cfg['lamb_min'], a_cfg['lamb_max'],
                                                            weight_decay=config.weight_decay)
                                    losses.append(angular_loss)
                                    insert_dict('aloss', angular_loss)
                                # Split Loss
                                if 'split' in config.losses.keys():
                                    split_losses = tflib.split_softmax(prelogits, labels, num_classes, 
                                                            global_step, gamma=config.losses['split']['gamma'], 
                                                            weight_decay=config.weight_decay)
                                    losses.extend(split_losses)
                                    insert_dict('loss', split_losses[0])

                               # Collect all losses
                                reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), name='reg_loss')
                                losses.append(reg_loss)
                                insert_dict('reg_loss', reg_loss)

                                total_loss = tf.add_n(losses, name='total_loss')
                                grads_split = tf.gradients(total_loss, tf.trainable_variables())
                                grads_splits.append(grads_split)



                # Merge the splits
                grads = tflib.average_grads(grads_splits)
                for k,v in split_dict.items():
                    v = tflib.average_tensors(v)
                    split_dict[k] = v
                    if 'loss' in k:
                        tf.summary.scalar('losses/' + k, v)
                    else:
                        tf.summary.scalar(k, v)


                # Training Operaters
                apply_gradient_op = tflib.apply_gradient(tf.trainable_variables(), grads, config.optimizer,
                                        learning_rate_placeholder, config.learning_rate_multipliers)

                update_global_step_op = tf.assign_add(global_step, 1)

                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

                train_ops = [apply_gradient_op, update_global_step_op] + update_ops
                train_op = tf.group(*train_ops)

                tf.summary.scalar('learning_rate', learning_rate_placeholder)
                summary_op = tf.summary.merge_all()

                # Initialize variables
                self.sess.run(tf.local_variables_initializer())
                self.sess.run(tf.global_variables_initializer())
                self.saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=None)

                # Keep useful tensors
                self.image_batch_placeholder = image_batch_placeholder
                self.label_batch_placeholder = label_batch_placeholder 
                self.learning_rate_placeholder = learning_rate_placeholder 
                self.keep_prob_placeholder = keep_prob_placeholder 
                self.phase_train_placeholder = phase_train_placeholder 
                self.global_step = global_step
                self.watch_list = split_dict
                self.train_op = train_op
                self.summary_op = summary_op
                


    def train(self, image_batch, label_batch, learning_rate, keep_prob):
        feed_dict = {self.image_batch_placeholder: image_batch,
                    self.label_batch_placeholder: label_batch,
                    self.learning_rate_placeholder: learning_rate,
                    self.keep_prob_placeholder: keep_prob,
                    self.phase_train_placeholder: True,}
        _, wl, sm = self.sess.run([self.train_op, self.watch_list, self.summary_op], feed_dict = feed_dict)
        step = self.sess.run(self.global_step)

        return wl, sm, step
    
    def restore_model(self, model_dir, restore_scopes):
        with self.graph.as_default():
            trainable_variables = tf.trainable_variables()
            tflib.restore_model(self.sess, trainable_variables, model_dir, restore_scopes)

    def save_model(self, model_dir, global_step):
        tflib.save_model(self.sess, self.saver, model_dir, global_step)
        

    def load_model(self, *args):
        tflib.load_model(self.sess, *args)
        self.phase_train_placeholder = self.graph.get_tensor_by_name('phase_train:0')
        self.keep_prob_placeholder = self.graph.get_tensor_by_name('keep_prob:0')
        self.inputs = self.graph.get_tensor_by_name('inputs:0')
        self.outputs = self.graph.get_tensor_by_name('outputs:0')

    def extract_feature(self, images, batch_size, preprocess=False, config=None, is_training=False):
        num_images = images.shape[0] if type(images)==np.ndarray else len(images)
        num_features = self.outputs.shape[1]
        result = np.ndarray((num_images, num_features), dtype=np.float32)
        for start_idx in range(0, num_images, batch_size):
            end_idx = min(num_images, start_idx + batch_size)
            inputs = images[start_idx:end_idx]
            if preprocess:
                assert config is not None
                inputs = utils.preprocess(inputs, config, is_training)
            feed_dict = {self.inputs: inputs,
                        self.phase_train_placeholder: False,
                        self.keep_prob_placeholder: 1.0}
            result[start_idx:end_idx] = self.sess.run(self.outputs, feed_dict=feed_dict)
        return result

        
