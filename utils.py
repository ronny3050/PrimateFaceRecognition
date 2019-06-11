"""Utilities for training and testing
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

import sys
import os
import numpy as np
from scipy import misc
import imp
import time
import math
import random
from datetime import datetime
import shutil
# from threading import Thread
# from Queue import Queue
from multiprocessing import Process, Queue
import h5py
import facepy
from sklearn.model_selection import KFold

def import_file(full_path_to_module, name='module.name'):
    if full_path_to_module is None:
        print('Configuration file not added as an argument')
        sys.exit(1)
    if not os.path.isfile(full_path_to_module):
        print('{} does not exist. Please check.'.format(full_path_to_module))
        sys.exit(1)
    module_obj = imp.load_source(name, full_path_to_module)
    
    return module_obj

def create_log_dir(config, config_file):
    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    log_dir = os.path.join(os.path.expanduser(config.log_base_dir), config.name, subdir)
    if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir)
    shutil.copyfile(config_file, os.path.join(log_dir,'config.py'))

    return log_dir

class DataClass():
    def __init__(self, class_name, indices, label):
        self.class_name = class_name
        self.indices = list(indices)
        self.label = label
        return

    def get_samples(self, num_samples_per_class, exception=None):
        indices_temp = self.indices[:]
        if exception is not None:
            indices_temp.remove(exception)
        indices = []
        iterations = int(np.ceil(1.0*num_samples_per_class / len(indices_temp)))
        # iterations = 1
        for i in range(iterations):
            sample_indices = np.random.permutation(indices_temp)
            indices.append(sample_indices)
        indices = np.concatenate(indices, axis=0)[:num_samples_per_class]
        # indices = indices[:min(indices.size, num_samples_per_class)]
        return indices

    def build_clusters(self, cluster_size):
        permut_indices = np.random.permutation(self.indices)
        cutoff = (permut_indices.size // cluster_size) * cluster_size
        clusters = np.reshape(permut_indices[:cutoff], [-1, cluster_size])
        clusters = list(clusters)
        if permut_indices.size > cutoff:
            last_cluster = permut_indices[cutoff:]
            clusters.append(last_cluster)
        return clusters

    def cutoff_samples(self, num_samples):
        cutoff = min(len(self.indices), num_samples)
        self.indices = self.indices[:cutoff]

class Dataset():

    def __init__(self, path=None):
        self.num_classes = None
        self.classes = None
        self.images = None
        self.labels = None
        self.index_queue = None
        self.queue_idx = None
        self.cluster_queue = None
        self.cluster_queue_idx = None
        self.batch_queue = None
        self.class_weights = None

        if path is not None:
            self.init_from_path(path)

    def clear(self):
        del self.classes
        self.__init__()

    def init_from_path(self, path):
        path = os.path.expanduser(path)
        _, ext = os.path.splitext(path)
        if os.path.isdir(path):
            self.init_from_folder(path)
        elif ext == '.txt':
            self.init_from_list(path)
        elif ext == '.hdf5':
            self.init_from_hdf5(path)
        else:
            raise ValueError('Cannot initialize dataset from path: %s\n\
                It should be either a folder, .txt or .hdf5 file' % path)
        print('%d images of %d classes loaded' % (len(self.images), self.num_classes))

    def init_from_folder(self, folder):
        folder = os.path.expanduser(folder)
        class_names = os.listdir(folder)
        class_names.sort()
        classes = []
        images = []
        labels = []
        for label, class_name in enumerate(class_names):
            classdir = os.path.join(folder, class_name)
            if os.path.isdir(classdir):
                images_class = os.listdir(classdir)
                images_class = [os.path.join(classdir,img) for img in images_class]
                indices_class = np.arange(len(images), len(images) + len(images_class))
                classes.append(DataClass(class_name, indices_class, label))
                images.extend(images_class)
                labels.extend(len(images_class) * [label])
        self.classes = np.array(classes, dtype=np.object)
        self.images = np.array(images, dtype=np.object)
        self.labels = np.array(labels, dtype=np.int32)
        self.num_classes = len(classes)

    def init_from_list(self, filename):
        with open(filename, 'r') as f:
            lines = f.readlines()
        lines = [line.strip().split(' ') for line in lines]
        assert len(lines)>0 and len(lines[0])==2, \
            'List file must be in format: "fullpath(str) label(int)"'
        images = [line[0] for line in lines]
        labels = [int(line[1]) for line in lines]
        self.images = np.array(images, dtype=np.object)
        self.labels = np.array(labels, dtype=np.int32)
        self.init_classes()


    def init_from_hdf5(self, filename):
        with h5py.File(filename, 'r') as f:
            self.images = np.array(f['images'])
            self.labels = np.array(f['labels'])
        self.init_classes()


    def init_classes(self):
        dict_classes = {}
        classes = []
        for i, label in enumerate(self.labels):
            if not label in dict_classes:
                dict_classes[label] = [i]
            else:
                dict_classes[label].append(i)
        for label, indices in dict_classes.items():
            classes.append(DataClass(str(label), indices, label))
        self.classes = np.array(classes, dtype=np.object)
        self.num_classes = len(classes)
       

    def build_subset_from_classes(self, classes):
        images = []
        labels = []
        classes_new = []
        for i, c in enumerate(classes):
            n = len(c.indices)
            images.extend(self.images[c.indices])
            labels.extend([i] * n)
        subset = Dataset()
        subset.images = np.array(images, dtype=np.object)
        subset.labels = np.array(labels, dtype=np.int32)
        subset.init_classes()
        subset.num_classes = len(classes)

        print('built subset: %d images of %d classes' % (len(subset.images), subset.num_classes))
        return subset

    def split_k_folds(self, k):
        self.kf = KFold(n_splits = k)
        self.train_idx = []
        self.test_idx = []
        for train, test in self.kf.split(self.classes):
            self.train_idx.append(train)
            self.test_idx.append(test)



    def get_fold(self, fold):
        k = self.kf.get_n_splits(self.classes)
        assert fold <= k
        # Concatenate the classes in difference folds for trainset
        #trainset_classes = [c for i in range(k) if i!=fold for c in self.k_folds_classes[i]]
        #testset_classes = self.k_folds_classes[fold]
        trainset = self.build_subset_from_classes(self.classes[self.train_idx[fold-1]])
        testset = self.build_subset_from_classes(self.classes[self.test_idx[fold-1]])
        return trainset, testset


    def init_index_queue(self, random=True):
        size = self.images.shape[0]
        if random:
            self.index_queue = np.random.permutation(size)
        else:
            self.index_queue = np.arange(size)
        self.queue_idx = 0

    def pop_index_queue(self, num, random=True):
        if self.index_queue is None:
            self.init_index_queue(random)
        result = []
        while num >= len(self.index_queue) - self.queue_idx:
            result.extend(self.index_queue[self.queue_idx:])
            num -= len(self.index_queue) - self.queue_idx
            self.init_index_queue(random)
            self.queue_idx = 0
        result.extend(self.index_queue[self.queue_idx : self.queue_idx+num])
        self.queue_idx += num
        return result

    def init_cluster_queue(self, cluster_size):
        assert type(cluster_size) == int
        self.cluster_queue = []
        for dataclass in self.classes:
            self.cluster_queue.extend(dataclass.build_clusters(cluster_size))
        random.shuffle(self.cluster_queue)
    
        self.cluster_queue = [idx for cluster in self.cluster_queue for idx in cluster]
        self.cluster_queue_idx = 0

        
    def pop_cluster_queue(self, num_clusters, cluster_size):
        if self.cluster_queue is None:
            self.init_cluster_queue(cluster_size)
        result = []
        while num_clusters >= len(self.cluster_queue) - self.cluster_queue_idx:
            result.extend(self.cluster_queue[self.cluster_queue_idx:])
            num_clusters -= len(self.cluster_queue) - self.cluster_queue_idx
            self.init_cluster_queue(cluster_size)
        result.extend(self.cluster_queue[self.cluster_queue_idx : self.cluster_queue_idx+num_clusters])
        self.cluster_queue_idx += num_clusters
        return result

    def get_batch(self, batch_size):
        indices_batch = self.pop_index_queue(batch_size, True)

        image_batch = self.images[indices_batch]
        label_batch = self.labels[indices_batch]
        return image_batch, label_batch

    def get_batch_classes(self, batch_size, num_classes_per_batch):
        # classes_batch = np.random.permutation(self.num_classes)[:num_classes_per_batch]
        # classes_batch = self.sample_classes_by_weight(num_classes_per_batch)

        #indices_root = self.pop_index_queue(num_classes_per_batch)
        #classes_batch = self.labels[indices_root]

        assert batch_size % num_classes_per_batch == 0
        num_samples_per_class = int(batch_size / num_classes_per_batch)

        indices_batch = self.pop_cluster_queue(batch_size, num_samples_per_class)

        #ndices_batch = [indices_root]
        #for i, class_id in enumerate(classes_batch):
        #    indices_batch.append(self.classes[class_id].get_samples(num_samples_per_class, indices_root[i]))
           
        # indices_batch = np.concatenate(indices_batch, axis=0)
        image_batch = self.images[indices_batch]
        label_batch = self.labels[indices_batch]
        return image_batch, label_batch



    def sample_classes_by_weight(self, num_classes_per_batch):
        if self.class_weights is None:
            self.class_weights = [len(dataclass.indices) for dataclass in self.classes]
            self.class_weights = np.array(self.class_weights, dtype=np.float32)
            self.class_weights = np.square(self.class_weights)
            self.class_weights = self.class_weights / np.sum(self.class_weights)
        p = self.class_weights
        selected = []
        for i in range(num_classes_per_batch):
            # p[selected] = 0.0
            # p = p / np.sum(p)
            select = np.random.choice(p.size, p=p)
            selected.append(select)
        return selected

    def get_samples_per_class(self, num_samples_per_class):
        indices = []
        for data_class in self.classes:
            indices.append(data_class.get_samples(num_samples_per_class))
        indices = np.concatenate(indices, axis=0)
        images = self.images[indices]
        labels = self.labels[indices]
        return images, labels

    # Multithreading preprocessing images
    def start_batch_queue(self, config, is_training, maxsize=16, num_threads=1):
        self.batch_queue = Queue(maxsize=maxsize)
        def batch_queue_worker():
            while True:
                if config.template_batch:
                    image_path_batch, label_batch = \
                        self.get_batch_classes(config.batch_size, config.num_classes_per_batch)
                else:
                    image_path_batch, label_batch = self.get_batch(config.batch_size)
                image_batch = preprocess(image_path_batch, config, is_training)
                self.batch_queue.put((image_batch, label_batch))

        for i in range(num_threads):
            worker = Process(target=batch_queue_worker)
            worker.daemon = True
            worker.start()
    
    
    def pop_batch_queue(self):
        batch = self.batch_queue.get(block=True, timeout=60)
        return batch



# Calulate the shape for creating new array given (w,h)
def get_new_shape(images, size):
    w, h = tuple(size)
    shape = list(images.shape)
    shape[1] = h
    shape[2] = w
    shape = tuple(shape)
    return shape

def random_crop(images, size):
    n, _h, _w = images.shape[:3]
    w, h = tuple(size)
    shape_new = get_new_shape(images, size)
    assert (_h>=h and _w>=w)

    images_new = np.ndarray(shape_new, dtype=images.dtype)

    y = np.random.randint(low=0, high=_h-h+1, size=(n))
    x = np.random.randint(low=0, high=_w-w+1, size=(n))

    for i in range(n):
        images_new[i] = images[i, y[i]:y[i]+h, x[i]:x[i]+w]

    return images_new

def center_crop(images, size):
    n, _h, _w = images.shape[:3]
    w, h = tuple(size)
    assert (_h>=h and _w>=w)

    y = int(round(0.5 * (_h - h)))
    x = int(round(0.5 * (_w - w)))

    images_new = images[:, y:y+h, x:x+w]

    return images_new

def random_flip(images):
    images_new = images
    flips = np.random.rand(images_new.shape[0])>=0.5
    
    for i in range(images_new.shape[0]):
        if flips[i]:
            images_new[i] = np.fliplr(images[i])

    return images_new

def resize(images, size):
    n, _h, _w = images.shape[:3]
    w, h = tuple(size)
    shape_new = get_new_shape(images, size)

    images_new = np.ndarray(shape_new, dtype=images.dtype)

    for i in range(n):
        images_new[i] = misc.imresize(images[i], (h,w))

    return images_new

def standardize_images(images, standard):
    if standard=='mean_scale':
        mean = 127.5
        std = 128.0
    elif standard=='scale':
        mean = 0.0
        std = 255.0
    elif standard=='deb':
        mean = np.mean(images,axis=(1,2,3)).reshape([-1,1,1,1])
        std = np.std(images, axis=(1,2,3)).reshape([-1,1,1,1])
    images_new = images.astype(np.float32)
    images_new = (images_new - mean) / std
    return images_new

def random_downsample(images, min_ratio):
    n, _h, _w = images.shape[:3]
    images_new = images
    ratios = min_ratio + (1-min_ratio) * np.random.rand(images_new.shape[0])

    for i in range(images_new.shape[0]):
        w = int(round(ratios[i] * _w))
        h = int(round(ratios[i] * _h))
        images_new[i,:h,:w] = misc.imresize(images[i], (h,w))
        images_new[i] = misc.imresize(images_new[i,:h,:w], (_h,_w))
        
    return images_new


def preprocess(images, config, is_training=False):
    # Load images first if they are file paths
    if type(images[0]) == str:
        image_paths = images
        images = []
        assert (config.channels==1 or config.channels==3)
        mode = 'RGB' if config.channels==3 else 'I'
        print('Reading images ...')
        for idx, image_path in enumerate(image_paths):
                images.append(misc.imread(image_path, mode=mode))
        print('Done reading images ... ')
            
        images = np.stack(images, axis=0)

    # Process images
    f = {
        'resize': resize,
        'random_crop': random_crop,
        'center_crop': center_crop,
        'random_flip': random_flip,
        'standardize': standardize_images,
        'random_downsample': random_downsample,
    }
    proc_funcs = config.preprocess_train if is_training else config.preprocess_test

    for name, args in proc_funcs:
        images = f[name](images, *args)
    if len(images.shape) == 3:
        images = images[:,:,:,None]
    return images
        
def get_updated_learning_rate(global_step, config):
    if config.learning_rate_strategy == 'step':
        max_step = -1
        learning_rate = 0.0
        for step, lr in config.learning_rate_schedule.items():
            if global_step >= step and step > max_step:
                learning_rate = lr
                max_step = step
        if max_step == -1:
            raise ValueError('cannot find learning rate for step %d' % global_step)
    elif config.learning_rate_strategy == 'cosine':
        initial = config.learning_rate_schedule['initial']
        interval = config.learning_rate_schedule['interval']
        end_step = config.learning_rate_schedule['end_step']
        step = math.floor(float(global_step) / interval) * interval
        assert step <= end_step
        learning_rate = initial * 0.5 * (math.cos(math.pi * step / end_step) + 1)
    return learning_rate

def display_info(epoch, step, duration, watch_list):
    sys.stdout.write('[%d][%d] time: %2.2f' % (epoch+1, step+1, duration))
    for item in watch_list.items():
        if type(item[1]) in [np.float32, np.float64]:
            sys.stdout.write('   %s: %2.3f' % (item[0], item[1]))
        elif type(item[1]) in [np.int32, np.int64, np.bool]:
            sys.stdout.write('   %s: %d' % (item[0], item[1]))
    sys.stdout.write('\n')

def get_pairwise_score_label(score_mat, label):
    n = label.size
    assert score_mat.shape[0]==score_mat.shape[1]==n
    triu_indices = np.triu_indices(n, 1)
    if len(label.shape)==1:
        label = label[:, None]
    label_mat = label==label.T
    score_vec = score_mat[triu_indices]
    label_vec = label_mat[triu_indices]
    return score_vec, label_vec

def split_batches_and_exec(input, batch_size, batch_func):
    '''Split the input into batches to execute given functions.

    All results are assumed to be arrays and will be merged along axis 0.
    '''
    length = input.shape[0] if type(input)==np.ndarray else len(input)
    
    results = []
    for start_idx in range(0, length, batch_size):
        end_idx = min(length, start_idx + batch_size)
        results.append(batch_func(input[start_idx:end_idx]))

    result = np.concatenate(results, axis=0)

    return result

def test_roc(features, labels, FARs):
    n, d = features.shape
    assert n % 2 == 0
    features = np.reshape(features, [-1,2,d])
    feat1, feat2 = features[:,0,:], features[:,1,:]
    score_mat = - facepy.metric.euclidean(feat1, feat2)
    label_mat = np.eye(n//2, dtype=np.bool)

    TARs, FARs, thresholds = facepy.evaluation.ROC(score_mat.flatten(), label_mat.flatten(), FARs=FARs)

    return TARs, FARs, thresholds
