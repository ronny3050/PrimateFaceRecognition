"""Functions for building tensorflow graph
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

import os
import tensorflow as tf
import tensorflow.contrib.slim as slim

def average_tensors(tensors, name=None):
    if len(tensors) == 1:
        return tf.identity(tensors[0], name=name)
    else:
        # Each tensor in the list should be of the same size
        expanded_tensors = []

        for t in tensors:
            expanded_t = tf.expand_dims(t, 0)
            expanded_tensors.append(expanded_t)

        average_tensor = tf.concat(axis=0, values=expanded_tensors)
        average_tensor = tf.reduce_mean(average_tensor, 0, name=name)

        return average_tensor


def average_grads(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
        tower_grads: List of lists of gradients. The outer list is over different 
        towers. The inner list is over the gradient calculation in each tower.
    Returns:
        List of gradients where the gradient has been averaged across all towers.
    """
    if len(tower_grads) == 1:
        return tower_grads[0]
    else:
        average_grads = []
        for grad_ in zip(*tower_grads):
            # Note that each grad looks like the following:
            #   (grad0_gpu0, ... , grad0_gpuN)
            average_grad = None if grad_[0]==None else average_tensors(grad_)
            average_grads.append(average_grad)

        return average_grads

def apply_gradient(update_gradient_vars, grads, optimizer, learning_rate, learning_rate_multipliers=None):
    assert(len(grads)==len(update_gradient_vars))
    if learning_rate_multipliers is None: learning_rate_multipliers = {}
    # Build a dictionary to save multiplier config
    # format -> {scope_name: ((grads, vars), lr_multi)}
    learning_rate_dict = {}
    learning_rate_dict['__default__'] = ([], 1.0)
    for scope, multiplier in learning_rate_multipliers.items():
        assert scope != '__default__'
        learning_rate_dict[scope] = ([], multiplier)

    # Scan all the variables, insert into dict
    scopes = learning_rate_dict.keys()
    for var, grad in zip(update_gradient_vars, grads):
        count = 0
        scope_temp = ''
        for scope in scopes:
            if scope in var.name:
                scope_temp = scope
                count += 1
        assert count <= 1, "More than one multiplier scopes appear in variable: %s" % var.name
        if count == 0: scope_temp = '__default__'
        if grad is not None:
            learning_rate_dict[scope_temp][0].append((grad, var))
     
    # Build a optimizer for each multiplier scope
    apply_gradient_ops = []
    print('\nLearning rate multipliers:')
    for scope, scope_content in learning_rate_dict.items():
        scope_grads_vars, multiplier = scope_content
        print('%s:\n  # variables: %d\n  lr_multi: %f' % (scope, len(scope_grads_vars), multiplier))
        if len(scope_grads_vars) == 0:
            continue
        scope_learning_rate = multiplier * learning_rate
        if optimizer=='ADAGRAD':
            opt = tf.train.AdagradOptimizer(scope_learning_rate)
        elif optimizer=='ADADELTA':
            opt = tf.train.AdadeltaOptimizer(scope_learning_rate, rho=0.9, epsilon=1e-6)
        elif optimizer=='ADAM':
            opt = tf.train.AdamOptimizer(scope_learning_rate, beta1=0.9, beta2=0.999, epsilon=0.1)
        elif optimizer=='RMSPROP':
            opt = tf.train.RMSPropOptimizer(scope_learning_rate, decay=0.9, momentum=0.9, epsilon=1.0)
        elif optimizer=='MOM':
            opt = tf.train.MomentumOptimizer(scope_learning_rate, 0.9, use_nesterov=False)
        elif optimizer=='SGD':
            opt = tf.train.GradientDescentOptimizer(scope_learning_rate)
        else:
            raise ValueError('Invalid optimization algorithm')
        apply_gradient_ops.append(opt.apply_gradients(scope_grads_vars))
    print('')
    apply_gradient_op = tf.group(*apply_gradient_ops)

    return apply_gradient_op


def rank_accuracy(logits, label, batch_size, k=1):
    _, arg_top = tf.nn.top_k(logits, k)
    label = tf.cast(label, tf.int32)
    label = tf.reshape(label, [batch_size, 1])
    label = tf.tile(label, [1, k])
    correct = tf.reduce_any(tf.equal(label, arg_top), axis=1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    return accuracy



def collect_watch_list():
    '''Collect the variables in watch list.

    Tensors or Varialbes can be addto collection 'watchlist' with 
    the type of tuple ('name', var/tensor). The functions collects
    them and returns a dict for evaluate
    '''
    watch_list = {}
    for pair in tf.get_collection('watch_list'):
        watch_list[pair[0]] = pair[1]
    return watch_list


def save_model(sess, saver, model_dir, global_step):
    with sess.graph.as_default():
            checkpoint_path = os.path.join(model_dir, 'ckpt')
            metagraph_path = os.path.join(model_dir, 'graph.meta')

            print('Saving variables...')
            saver.save(sess, checkpoint_path, global_step=global_step, write_meta_graph=False)
            if not os.path.exists(metagraph_path):
                print('Saving metagraph...')
                saver.export_meta_graph(metagraph_path)

def restore_model(sess, var_list, model_dir, restore_scopes=None):
    ''' Load the variable values from a checkpoint file into pre-defined graph.
    Filter the variables so that they contain at least one of the given keywords.'''
    with sess.graph.as_default():
            if restore_scopes is not None:
                var_list = [var for var in var_list if any([scope in var.name for scope in restore_scopes])]
            
            model_dir = os.path.expanduser(model_dir)
            ckpt_file = tf.train.latest_checkpoint(model_dir)

            print('Restoring variables from %s ...' % ckpt_file)
            saver = tf.train.Saver(var_list)
            saver.restore(sess, ckpt_file)

def load_model(sess, model_path, scope=None):
    ''' Load the the graph and variables values from a model path.
    Model path is either a a frozen graph or a directory with both
    a .meta file and checkpoint files.'''
    with sess.graph.as_default():
            model_path = os.path.expanduser(model_path)
            if (os.path.isfile(model_path)):
                # Frozen grpah
                print('Model filename: %s' % model_path)
                with gfile.FastGFile(model_path,'rb') as f:
                    graph_def = tf.GraphDef()
                    graph_def.ParseFromString(f.read())
                    tf.import_graph_def(graph_def, name='')
            else:
                # Load grapha and variables separatedly.
                meta_files = [file for file in os.listdir(model_path) if file.endswith('.meta')]
                assert len(meta_files) == 1
                meta_file = os.path.join(model_path, meta_files[0])
                ckpt_file = tf.train.latest_checkpoint(model_path)
                
                print('Metagraph file: %s' % meta_file)
                print('Checkpoint file: %s' % ckpt_file)
                saver = tf.train.import_meta_graph(meta_file, clear_devices=True,  import_scope=scope)
                saver.restore(sess, ckpt_file)


def euclidean_distance(X, Y, sqrt=False):
    '''Compute the distance between each X and Y.

    Args: 
        X: a (m x d) tensor
        Y: a (d x n) tensor
    
    Returns: 
        diffs: an m x n distance matrix.
    '''
    with tf.name_scope('EuclideanDistance'):
        XX = tf.reduce_sum(tf.square(X), 1, keep_dims=True)
        YY = tf.reduce_sum(tf.square(Y), 0, keep_dims=True)
        XY = tf.matmul(X, Y)
        diffs = XX + YY - 2*XY
        if sqrt == True:
            diffs = tf.sqrt(tf.maximum(0.0, diffs))
    return diffs

def cosine_softmax(prelogits, label, num_classes, weight_decay, gamma=16.0, reuse=None):
    
    nrof_features = prelogits.shape[1].value
    
    with tf.variable_scope('Logits', reuse=reuse):
        weights = tf.get_variable('weights', shape=(nrof_features, num_classes),
                regularizer=slim.l2_regularizer(weight_decay),
                initializer=slim.xavier_initializer(),
                # initializer=tf.truncated_normal_initializer(stddev=0.1),
                dtype=tf.float32)
        alpha = tf.get_variable('alpha', shape=(),
                regularizer=slim.l2_regularizer(1e-2),
                initializer=tf.constant_initializer(1.00),
                trainable=True,
                dtype=tf.float32)

        weights_normed = tf.nn.l2_normalize(weights, dim=0)
        prelogits_normed = tf.nn.l2_normalize(prelogits, dim=1)

        if gamma == 'auto':
            gamma = tf.nn.softplus(alpha)
        else:
            assert type(gamma) == float
            gamma = tf.constant(gamma)

        logits = gamma * tf.matmul(prelogits_normed, weights_normed)


    cross_entropy =  tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(\
            labels=label, logits=logits), name='cross_entropy')

    tf.summary.scalar('gamma', gamma)
    tf.add_to_collection('watch_list', ('gamma', gamma))

    return logits, cross_entropy

def norm_loss(prelogits, alpha, reuse=None):
    with tf.variable_scope('NormLoss', reuse=reuse):
        sigma = tf.get_variable('sigma', shape=(),
            # regularizer=slim.l2_regularizer(weight_decay),
            initializer=tf.constant_initializer(0.1),
            trainable=True,
            dtype=tf.float32)
    prelogits_norm = tf.reduce_sum(tf.square(prelogits), axis=1)
    # norm_loss = alpha * tf.square(tf.sqrt(prelogits_norm) - sigma)
    norm_loss = alpha * prelogits_norm
    norm_loss = tf.reduce_mean(norm_loss, axis=0, name='norm_loss')

    # tf.summary.scalar('sigma', sigma)
    # tf.add_to_collection('watch_list', ('sigma', sigma))
    return norm_loss

def angular_softmax(prelogits, label, num_classes, global_step, 
            m, lamb_min, lamb_max, weight_decay, reuse=None):
    num_features = prelogits.shape[1].value
    batch_size = tf.shape(prelogits)[0]
    lamb_min = lamb_min
    lamb_max = lamb_max
    lambda_m_theta = [
        lambda x: x**0,
        lambda x: x**1,
        lambda x: 2.0*(x**2) - 1.0,
        lambda x: 4.0*(x**3) - 3.0*x,
        lambda x: 8.0*(x**4) - 8.0*(x**2) + 1.0,
        lambda x: 16.0*(x**5) - 20.0*(x**3) + 5.0*x
    ]

    with tf.variable_scope('AngularSoftmax', reuse=reuse):
        weights = tf.get_variable('weights', shape=(num_features, num_classes),
                regularizer=slim.l2_regularizer(1e-4),
                initializer=slim.xavier_initializer(),
                # initializer=tf.truncated_normal_initializer(stddev=0.1),
                trainable=True,
                dtype=tf.float32)
        lamb = tf.get_variable('lambda', shape=(),
                initializer=tf.constant_initializer(lamb_max),
                trainable=False,
                dtype=tf.float32)
        prelogits_norm  = tf.sqrt(tf.reduce_sum(tf.square(prelogits), axis=1, keep_dims=True))
        weights_normed = tf.nn.l2_normalize(weights, dim=0)
        prelogits_normed = tf.nn.l2_normalize(prelogits, dim=1)

        # Compute cosine and phi
        cos_theta = tf.matmul(prelogits_normed, weights_normed)
        cos_theta = tf.minimum(1.0, tf.maximum(-1.0, cos_theta))
        theta = tf.acos(cos_theta)
        cos_m_theta = lambda_m_theta[m](cos_theta)
        k = tf.floor(m*theta / 3.14159265)
        phi_theta = tf.pow(-1.0, k) * cos_m_theta - 2.0 * k

        cos_theta = cos_theta * prelogits_norm
        phi_theta = phi_theta * prelogits_norm

        lamb_new = tf.maximum(lamb_min, lamb_max/(1.0+0.1*tf.cast(global_step, tf.float32)))
        update_lamb = tf.assign(lamb, lamb_new)
        
        # Compute loss
        with tf.control_dependencies([update_lamb]):
            label_dense = tf.one_hot(label, num_classes, dtype=tf.float32)

            logits = cos_theta
            logits -= label_dense * cos_theta * 1.0 / (1.0+lamb)
            logits += label_dense * phi_theta * 1.0 / (1.0+lamb)
            
            cross_entropy =  tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(\
                labels=label, logits=logits), name='cross_entropy')

        tf.add_to_collection('watch_list', ('lamb', lamb))

    return cross_entropy


def split_softmax(prelogits, label, num_classes, 
                global_step, weight_decay, gamma=16.0, reuse=None):
    nrof_features = prelogits.shape[1].value
    batch_size = tf.shape(prelogits)[0]
    with tf.variable_scope('SplitSoftmax', reuse=reuse):
        weights = tf.get_variable('weights', shape=(num_classes, nrof_features),
                regularizer=slim.l2_regularizer(weight_decay),
                initializer=slim.xavier_initializer(),
                # initializer=tf.truncated_normal_initializer(stddev=0.1),
                # initializer=tf.constant_initializer(0),
                trainable=True,
                dtype=tf.float32)
        alpha = tf.get_variable('alpha', shape=(),
                regularizer=slim.l2_regularizer(1e-2),
                initializer=tf.constant_initializer(1.00),
                trainable=True,
                dtype=tf.float32)
        beta = tf.get_variable('beta', shape=(),
                # regularizer=slim.l2_regularizer(1e-2),
                initializer=tf.constant_initializer(0.0),
                trainable=True,
                dtype=tf.float32)
        sigma = tf.get_variable('sigma', shape=(),
                regularizer=slim.l2_regularizer(1e-1),
                initializer=tf.constant_initializer(1.0),
                trainable=True,
                dtype=tf.float32)
        threshold_pos = tf.get_variable('threshold_pos', shape=(),
                initializer=tf.constant_initializer(16.0),
                trainable=False, 
                dtype=tf.float32)
        threshold_neg = tf.get_variable('threshold_neg', shape=(),
                initializer=tf.constant_initializer(0.0),
                trainable=False, 
                dtype=tf.float32)

        # Normalizing the vecotors
        weights_normed = tf.nn.l2_normalize(weights, dim=1)
        prelogits_normed = tf.nn.l2_normalize(prelogits, dim=1)
        # weights_normed = weights
        # prelogits_normed = prelogits

        # Caluculate Centers
        centers, label_center, center_idx, center_weight = centers_by_label(prelogits_normed, label)
        centers = tf.gather(centers, center_idx)
        centers_normed = tf.nn.l2_normalize(centers, dim=1)

        coef = 1.0
        # Label and logits between batch and examplars
        label_mat_glob = tf.one_hot(label, num_classes, dtype=tf.float32)
        label_mask_pos_glob = tf.cast(label_mat_glob, tf.bool)
        label_mask_neg_glob = tf.logical_not(label_mask_pos_glob)
        # label_exp_batch = tf.expand_dims(label, 1)
        # label_exp_glob = tf.expand_dims(label_history, 1)
        # label_mat_glob = tf.equal(label_exp_batch, tf.transpose(label_exp_glob))
        # label_mask_pos_glob = tf.cast(label_mat_glob, tf.bool)
        # label_mask_neg_glob = tf.logical_not(label_mat_glob)

        # dist_mat_glob = euclidean_distance(prelogits_normed, tf.transpose(weights_normed), False)
        dist_mat_glob = tf.matmul(prelogits_normed, tf.transpose(weights_normed)) # + beta
        dist_pos_glob = tf.boolean_mask(dist_mat_glob, label_mask_pos_glob)
        dist_neg_glob = tf.boolean_mask(dist_mat_glob, label_mask_neg_glob)

        logits_glob = coef * dist_mat_glob
        logits_pos_glob = tf.boolean_mask(logits_glob, label_mask_pos_glob)
        logits_neg_glob = tf.boolean_mask(logits_glob, label_mask_neg_glob)


        # Label and logits within batch
        label_exp_batch = tf.expand_dims(label, 1)
        label_mat_batch = tf.equal(label_exp_batch, tf.transpose(label_exp_batch))
        label_mask_pos_batch = tf.cast(label_mat_batch, tf.bool)
        label_mask_neg_batch = tf.logical_not(label_mask_pos_batch)
        mask_non_diag = tf.logical_not(tf.cast(tf.eye(batch_size), tf.bool))
        label_mask_pos_batch = tf.logical_and(label_mask_pos_batch, mask_non_diag)

        # dist_mat_batch = euclidean_distance(prelogits_normed, tf.transpose(prelogits_normed), False)
        dist_mat_batch = tf.matmul(prelogits_normed, tf.transpose(prelogits_normed))
        dist_pos_batch = tf.boolean_mask(dist_mat_batch, label_mask_pos_batch)
        dist_neg_batch = tf.boolean_mask(dist_mat_batch, label_mask_neg_batch)

        logits_batch =  coef * dist_mat_batch
        logits_pos_batch = tf.boolean_mask(logits_batch, label_mask_pos_batch)
        logits_neg_batch = tf.boolean_mask(logits_batch, label_mask_neg_batch)


        # num_anchor = 32
        # prelogits_anchor = tf.reshape(prelogits_normed[:num_anchor], [num_anchor, 1, nrof_features])
        # prelogits_refer = tf.reshape(prelogits_normed[num_anchor:], [num_anchor, -1, nrof_features])
        # dist_anchor = tf.reduce_sum(tf.square(prelogits_anchor-prelogits_refer), axis=2)
        # dist_anchor = tf.reshape(dist_anchor, [-1])
        # logits_anchor = -0.5 * gamma * dist_anchor
        

        logits_pos = logits_pos_glob
        logits_neg = logits_neg_glob
    
        dist_pos = dist_pos_glob
        dist_neg = dist_neg_glob

        # epsilon_trsd = 0.3
        t_pos = coef * (threshold_pos)
        t_neg = coef * (threshold_neg)


        if gamma == 'auto':
            # gamma = tf.nn.softplus(alpha)
            gamma = tf.log(tf.exp(1.0) + tf.exp(alpha))
        elif type(gamma) == tuple:
            t_min, decay = gamma
            epsilon = 1e-5
            t = t_min + 1.0/(epsilon + decay*tf.cast(global_step, tf.float32))
            gamma = 1.0 / t
        else:
            assert type(gamma) == float
            gamma = tf.constant(gamma)

        hinge_loss = lambda x: tf.nn.relu(1.0 + x)
        margin_func = hinge_loss

        # Losses
        losses = []
        # num_pos = tf.cast(0.95 * tf.cast(tf.size(logits_pos), tf.float32), tf.int32)
        # # num_neg = tf.cast(0.75 * tf.cast(tf.size(logits_neg), tf.float32), tf.int32)
        # q_d = tf.pow(tf.sqrt(dist_neg), 2-nrof_features)*tf.pow(1-0.25*dist_neg, (3-nrof_features)/2)
        # tf.add_to_collection('watch_list', ('q_d', tf.reduce_sum(q_d)))
        # q_d = tf.minimum(1.0, 1 * q_d / tf.reduce_sum(q_d))
        # tf.add_to_collection('watch_list', ('q_d', tf.reduce_mean(q_d)))
        # sample_mask = tf.random_uniform(shape=tf.shape(logits_neg)) <= q_d
        # sample_mask = logits_neg >= tf.reduce_min(logits_pos)
        # _logits_neg = tf.boolean_mask(logits_neg, sample_mask)
        # tf.add_to_collection('watch_list', ('sample_ratio', 
        #    tf.cast(tf.size(_logits_neg),tf.float32) / tf.cast(tf.size(logits_neg),tf.float32)))
               

        # gamma2 = 1 / 0.01
        _logits_pos = tf.reshape(logits_pos, [batch_size, -1])
        _logits_neg = tf.reshape(logits_neg, [batch_size, -1])

        norm = tf.square(tf.reduce_sum(tf.square(prelogits), axis=1, keep_dims=True))
        norm_weights = tf.norm(tf.gather(weights, label), axis=1, keep_dims=True)
        t_pos = (beta)
        t_neg = (beta)


        _logits_pos =  _logits_pos * gamma 
        _logits_neg =  _logits_neg * gamma
        # _logits_neg, _ = tf.nn.top_k(_logits_neg, num_neg)
        # _logits_pos, _ = tf.nn.top_k(_logits_pos, num_pos)
        # _logits_neg = tf.boolean_mask(_logits_neg, sample_mask)
        # _logits_pos = -tf.reduce_logsumexp(-_logits_pos)# , axis=1)[:,None]
        _logits_neg = tf.reduce_logsumexp(_logits_neg, axis=1)[:,None]
        # _logits_pos = tf.reduce_mean(_logits_pos)
        #-- Simulate Ranking
        # se_neg = tf.reduce_sum(tf.exp(_logits_neg))
        # min_pos = tf.reduce_min(_logits_pos)
        # t_pos = tf.stop_gradient(tf.log(se_neg))
        # t_neg = tf.stop_gradient(tf.log(se_neg - tf.exp(_logits_neg)))
        

        # norm = tf.reshape(prelogits[:,-1], [batch_size, -1])
        # norm_weighted = tf.exp(-norm)
        # norm_weighted = norm / tf.reduce_sum(norm) * tf.cast(tf.size(norm), tf.float32)

        # sigma_batch = tf.reshape(tf.gather(sigma, label), [batch_size, -1])

        m = 5.0
        # tf.add_to_collection('watch_list', ('m',m))

        factor = 1 / tf.cast(batch_size, tf.float32)
        bias = tf.log(tf.cast(num_classes, tf.float32))
        loss_pos = tf.nn.relu(m + _logits_neg - _logits_pos) * 0.5
        loss_neg = tf.nn.relu(m + _logits_neg - _logits_pos) * 0.5
        loss = tf.reduce_mean((loss_pos + loss_neg), name='split_loss')
        losses.extend([loss])
        tf.add_to_collection('watch_list', ('split_loss', loss))

        # Global loss
        # weights_batch = tf.gather(weights_normed, label)
        # _logits_pos_glob = tf.reduce_sum(tf.square(prelogits_normed - weights_batch), axis=1)  * coef * gamma
        _logits_pos_glob = tf.reshape(logits_pos_glob, [batch_size, -1]) * gamma
        _logits_neg_glob = tf.reshape(logits_neg_glob, [batch_size, -1]) * gamma
        _logits_neg_glob = tf.reduce_logsumexp(_logits_neg_glob) # , axis=1)[:,None]
        loss_glob = tf.reduce_mean(tf.nn.relu(1 + _logits_neg_glob - _logits_pos_glob), name='loss_glob')
        # losses.append(loss_glob)
        # tf.add_to_collection('watch_list', ('loss_glob', loss_glob))

        # Weight decay
        loss_weight = tf.reduce_sum( 1e-7 * tf.square(weights_normed), name='loss_weight')
        # losses.append(loss_weight)
        # tf.add_to_collection('watch_list', ('loss_weight', loss_weight))

        # Split Softmax
        # _logits_pos_glob = tf.reshape(logits_pos_glob, [batch_size, -1]) * gamma
        # _logits_neg_glob = tf.reshape(logits_neg_glob, [batch_size, -1]) * gamma
        # _logits_pos_glob = tf.log(tf.reduce_sum(tf.exp(_logits_pos_glob) + num_classes-1, axis=1)[:,None])
        # _logits_neg_glob = tf.reduce_logsumexp(_logits_neg_glob, axis=1)[:,None]
        # _t_pos = t_pos * gamma
        # _t_neg = t_neg * gamma
        # loss_pos = tf.reduce_mean(tf.nn.softplus(_t_pos - _logits_pos_glob), name='loss_pos')
        # loss_neg = tf.reduce_mean(tf.nn.softplus(_logits_neg_glob - _t_neg), name='loss_neg')
        # losses.extend([loss_pos, loss_neg])



        # Batch Center loss
        # centers_batch = tf.gather(centers, center_idx)
        centers_batch = tf.gather(weights_normed, label)
        dist_center = tf.reduce_sum(tf.square(prelogits_normed - centers_batch), axis=1)
        loss_center = tf.reduce_mean(1.0*dist_center, name='loss_center')
        # losses.append(loss_center)
        # tf.add_to_collection('watch_list', ('loss_center', loss_center))


        # Update threshold
        if not threshold_pos in tf.trainable_variables():
            # -- Mean threshold        
            mean_pos, var_pos = tf.nn.moments(dist_pos, axes=[0])
            mean_neg, var_neg = tf.nn.moments(dist_neg, axes=[0])
            std_pos = tf.sqrt(var_pos)
            std_neg = tf.sqrt(var_neg)
            threshold_batch = std_neg*mean_pos / (std_pos+std_neg) + std_pos*mean_neg / (std_pos+std_neg)
            threshold_pos_batch = threshold_neg_batch = threshold_batch
            # -- Logits
            # threshold_pos_batch = tf.reduce_logsumexp(_logits_neg)
            # threshold_neg_batch = -tf.reduce_logsumexp(-_logits_pos)
            # -- Quantile
            # diff_pos_sorted, _ = tf.nn.top_k(logits_pos, 2)
            # diff_neg_sorted, _ = tf.nn.top_k(logits_neg, 2704237)
            # threshold_pos_batch = diff_neg_sorted[-1]
            # threshold_neg_batch = diff_pos_sorted[-1]
            threshold_neg_batch = tf.reduce_min(_logits_pos)
            threshold_pos_batch = tf.reduce_max(_logits_neg)
            # -- Update
            diff_threshold_pos = threshold_pos - threshold_pos_batch
            diff_threshold_neg = threshold_neg - threshold_neg_batch
            diff_threshold_pos = 0.1 * diff_threshold_pos
            diff_threshold_neg = 0.1 * diff_threshold_neg
            threshold_pos_update_op = tf.assign_sub(threshold_pos, diff_threshold_pos)
            threshold_neg_update_op = tf.assign_sub(threshold_neg, diff_threshold_neg)
            threshold_update_op = tf.group(threshold_pos_update_op, threshold_neg_update_op)
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, threshold_update_op)


        # Update centers
        if not weights in tf.trainable_variables():
            weights_batch = tf.gather(weights, label)
            diff_centers = weights_batch - prelogits
            unique_label, unique_idx, unique_count = tf.unique_with_counts(label)
            appear_times = tf.gather(unique_count, unique_idx)
            appear_times = tf.reshape(appear_times, [-1, 1])
            diff_centers = diff_centers / tf.cast((1 + appear_times), tf.float32)
            diff_centers = 0.5 * diff_centers
            centers_update_op = tf.scatter_sub(weights, label, diff_centers)
            # centers_decay_op = tf.assign_sub(weights, 2*weight_decay*weights)# weight decay
            centers_update_op = tf.group(centers_update_op)
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, centers_update_op)

        # if not sigma in tf.trainable_variables(): 
        #     weights_batch = tf.gather(weights, label)
        #     diff_centers = weights_batch - prelogits
        #     _, var_pos = tf.nn.moments(diff_centers, axes=[0])
        #     sigma_batch = tf.reduce_mean(tf.sqrt(var_pos))
        #     diff_sigma = sigma - sigma_batch
        #     diff_sigma = 0.01 * diff_sigma
        #     sigma_update_op = tf.assign_sub(sigma, diff_sigma)
        #     tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, sigma_update_op)



        # Analysis
        mean_dist_pos = tf.reduce_mean(dist_pos, name='mean_dist_pos')
        mean_dist_neg = tf.reduce_mean(dist_neg, name='mean_dist_neg')
        acc_pos = tf.reduce_mean(tf.cast(tf.greater_equal(logits_pos, t_pos), tf.float32), name='acc_pos')
        acc_neg = tf.reduce_mean(tf.cast(tf.less(logits_neg, t_neg), tf.float32), name='acc_neg')
        tf.summary.scalar('threshold_pos', threshold_pos)
        tf.summary.scalar('mean_dist_pos', mean_dist_pos)
        tf.summary.scalar('mean_dist_neg', mean_dist_neg)
        tf.summary.scalar('acc_pos', acc_pos)
        tf.summary.scalar('acc_neg', acc_neg)
        tf.summary.scalar('gamma', gamma)
        tf.summary.scalar('alpha', alpha)
        tf.summary.scalar('beta', beta)
        tf.summary.histogram('dist_pos', dist_pos)
        tf.summary.histogram('dist_neg', dist_neg)
        # tf.summary.histogram('dist_neg_min', _logits_neg / coef)
        # tf.summary.histogram('sigma', sigma)

        # tf.add_to_collection('watch_list', ('alpha', alpha))
        tf.add_to_collection('watch_list', ('gamma', gamma))
        tf.add_to_collection('watch_list', ('alpha', alpha))
        tf.add_to_collection('watch_list', ('beta', beta))
        # tf.add_to_collection('watch_list', ('t_pos', t_pos))
        # tf.add_to_collection('watch_list', ('t_neg', tf.reduce_mean(t_neg)))
        # tf.add_to_collection('watch_list', ('dpos', mean_dist_pos))
        # tf.add_to_collection('watch_list', ('dneg', mean_dist_neg))
        # tf.add_to_collection('watch_list', ('loss_pos', loss_pos))
        # tf.add_to_collection('watch_list', ('loss_neg', loss_neg))
        # tf.add_to_collection('watch_list', ('sigma', sigma))
        # tf.add_to_collection('watch_list', ('logits_pos', tf.reduce_mean(_logits_pos)))
        # tf.add_to_collection('watch_list', ('logits_neg', tf.reduce_mean(_logits_neg)))
        # tf.add_to_collection('watch_list', ('acc_pos', acc_pos))
        # tf.add_to_collection('watch_list', ('acc_neg', acc_neg))

    return losses

def centers_by_label(features, label):
    # Compute centers within batch
    unique_label, unique_idx, unique_count = tf.unique_with_counts(label)
    num_centers = tf.size(unique_label)
    appear_times = tf.gather(unique_count, unique_idx)
    appear_times = tf.reshape(appear_times, [-1, 1])
    weighted_prelogits = features / tf.cast(appear_times, tf.float32)
    centers = tf.unsorted_segment_sum(weighted_prelogits, unique_idx, num_centers)
    return centers, unique_label, unique_idx, unique_count

def cluster_loss(prelogits, label, num_classes,
                weight_decay, gamma=16.0, reuse=None):
    embedding_size = prelogits.shape[1].value
    batch_size = tf.shape(prelogits)[0]
    with tf.variable_scope('ClusterLoss', reuse=reuse):
        alpha = tf.get_variable('alpha', shape=(),
                # regularizer=slim.l2_regularizer(weight_decay),
                initializer=tf.constant_initializer(1.0),
                trainable=True,
                dtype=tf.float32)
        gamma = gamma
        prelogits = tf.nn.l2_normalize(prelogits, dim=1) 
        centers, label_center, center_idx, center_weight = centers_by_label(prelogits, label)
        centers = tf.nn.l2_normalize(centers, dim=1)
        num_centers = tf.size(label_center)

        # Compute distance between centers
        dist_centers_mat = euclidean_distance(centers, tf.transpose(centers))
        mask_non_diag = tf.logical_not(tf.cast(tf.eye(num_centers), tf.bool))
        mask_triu = tf.cast(tf.matrix_band_part(tf.ones((num_centers, num_centers)), 0, -1), tf.bool)
        mask_triu = tf.logical_and(mask_non_diag, mask_triu)
        dist_centers_vec = tf.boolean_mask(dist_centers_mat, mask_triu)

        # Compute distance between instance and ceners
        centers_batch = tf.gather(centers, center_idx)
        dist_instance = euclidean_distance(prelogits, tf.transpose(centers))

        label_dense = tf.one_hot(center_idx, num_centers, dtype=tf.float32)
        label_pos = tf.cast(label_dense, tf.bool)
        label_neg = tf.logical_not(label_pos)
        dist_instance_pos = tf.boolean_mask(dist_instance, label_pos)
        dist_instance_neg = tf.boolean_mask(dist_instance, label_neg)



        # Losses
        alpha = 1.0
        gamma = 20.0
        dist_instance_pos = tf.reshape(dist_instance_pos, [batch_size, -1])
        dist_instance_neg = tf.reshape(dist_instance_neg, [batch_size, -1])
        logits_pos = - 0.5 * 2 * dist_instance_pos * gamma
        logits_neg = - 0.5 * dist_centers_vec * gamma
        # logits_pos = tf.reduce_mean(logits_pos)
        logits_neg = tf.reduce_logsumexp(logits_neg)#, axis=1)[:,None]
        # min_dist_centers = -tf.reduce_logsumexp(-dist_centers_vec)
        # loss_instance = tf.identity(alpha*dist_instance_pos - min_dist_centers)
        loss_instance = tf.reduce_mean(tf.nn.softplus(logits_neg - logits_pos))
        losses = [loss_instance]

        # Analysis
        tf.summary.histogram('prelogits', prelogits)
        # tf.summary.scalar('min_dist_centers', min_dist_centers)
        # tf.summary.histogram('min_dist_centers', min_dist_centers)
        tf.summary.histogram('dist_centers_vec', dist_centers_vec)
        tf.summary.histogram('dist_instances_pos', dist_instance_pos)
        # tf.add_to_collection('watch_list', ('dcenters', min_dist_centers))
        tf.add_to_collection('watch_list', ('loss', loss_instance))
        # tf.add_to_collection('watch_list', ('alpha', alpha))

    return losses

def binary_loss(prelogits, label, num_classes, 
                weight_decay, gamma=16.0, reuse=None):
    nrof_features = prelogits.shape[1].value
    batch_size = tf.shape(prelogits)[0]
    with tf.variable_scope('BinaryLoss', reuse=reuse):
        weights = tf.get_variable('weights', shape=(num_classes, nrof_features),
                # regularizer=slim.l2_regularizer(weight_decay),
                initializer=tf.truncated_normal_initializer(stddev=1.0),
                # initializer=tf.constant_initializer(1.0),
                trainable=True,
                dtype=tf.float32)

    weights_normed = tf.nn.sigmoid(weights)
    prelogits_normed = prelogits

    weights_batch = tf.gather(weights_normed, label)
    closs = tf.nn.sigmoid_cross_entropy_with_logits(logits=prelogits_normed, labels=weights_batch)
    closs = tf.reduce_sum(closs, axis=1)
    closs = tf.reduce_mean(closs, name='cross_entropy')

    p_pos = tf.reduce_mean(weights_normed, axis=0)
    p_neg = tf.reduce_mean(1-weights_normed, axis=0)
    eloss = (p_pos * tf.log(p_pos) + p_neg * tf.log(p_neg))
    eloss = tf.reduce_sum(eloss, name='entropy')

    losses = [closs, eloss]

    tf.add_to_collection('watch_list', ('closs', closs))
    tf.add_to_collection('watch_list', ('eloss', eloss))

    return losses
