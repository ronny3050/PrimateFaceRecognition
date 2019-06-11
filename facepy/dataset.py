"""Class definition for data structures.
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
import re
import numpy as np


from . import io

def copy_array(array):
    result_array = None
    if type(array) == list:
        if len(array) > 1:
            result_array = np.array(result_array)
        else:
            result_array = None
    elif type(array) == np.ndarray:
        result_array = array.copy()
    elif array is None:
        result_array = None
    else:
        raise ValueError('Invalid argument type: %s' % type(array))
    return result_array

def select(array, indices):
    if array is None or array.shape[0] == 0 \
        or indices is None or len(indices)==0:
        return None
    else:
        return array[indices]

class Template(object):
    def __init__(self, template_id=None, label=None, indices=None, dataset=None, features=None, images=None):
        self.id = template_id
        self.label = label
        self.indices = indices
        self._dataset = dataset # binded dataset
        self._features = features # static data to replace replace binded
        self._images =  images # static image to replace binded

        if self.indices is not None:
            self.indices = np.array(self.indices).reshape([-1])
        return

    def bindDataset(self, dataset):
        self._dataset = dataset
        return

    @property
    def features(self):
        if self._features is not None:
            return self._features
        elif self._dataset is not None:
            return select(self._dataset.features, self.indices)
        else:
            return None

    @features.setter
    def set_features(self, value):
        self._features = value

    @property
    def images(self):
        if self._images is not None:
            return self._images
        elif self._dataset is not None:
            return select(self._dataset.images, self.indices)
        else:
            return None

    @images.setter
    def set_images(self, value):
        self._images = value

class Dataset():
    def __init__(self, init_path=None, images=None, labels=None, 
                bboxes=None, landmarks=None, features=None, 
                folder_depth=None):
        self.images = copy_array(images)
        self.labels = copy_array(labels)
        self.bboxes = copy_array(bboxes)
        self.landmarks = copy_array(landmarks)
        self.features = copy_array(features)
        self.folder_depth = folder_depth

        if init_path:
            self.init_from_path(init_path)

        self.image_dict = None
        self.template_dict = None

    # Bind the templates with the dataset
    def bind_templates(self, templates, initialization=True):
        if initialization:
            self.template_dict = {}
        for template in templates:
            template.bindDataset(self)
            self.template_dict[template.id] = template
        return

    def get_templates(self, id_list):
        templates = []
        for template_id in id_list:
            templates.append(self.template_dict[template_id])
        return templates

    def get_template_pairs(self, pair_list):
        template_pairs = []
        for pair in pair_list:
            template1 = self.template_dict[pair[0]]
            template2 = self.template_dict[pair[1]]
            template_pairs.append((template1, template2))            
        return template_pairs

    def build_image_dict(self):
        assert type(self.images[0]) == str
        self.image_dict = {}
        for i, image in enumerate(self.images):
            if self.folder_depth is not None:
                image = str.join('/', re.split(r'/+', image)[-self.folder_depth:])
            self.image_dict[image] = i


    def find_images(self, images):
        indices = []
        if self.image_dict is None:
            self.build_image_dict()
        for image in images:
            if self.folder_depth is not None:
                image = str.join('/', re.split(r'/+', image)[-self.folder_depth:])
            assert image in self.image_dict
            indices.append(self.image_dict[image])
        return indices


    def subset(self, indices):
        return Dataset(images=select(self.images, indices), labels=select(self.labels, indices),
                    bboxes=select(self.bboxes, indices), landmarks=select(self.landmarks, indices),
                    features=select(self.features, indices), folder_depth=self.folder_depth)

    #### I/O ####
    def init_from_path(self, path):
        _, ext = os.path.splitext(path)
        if ext == '.txt':
            with open(path, 'r') as f:
                self.images = np.array([line.strip() for line in f.readlines()], dtype=np.object)
        else:
            raise ValueError('Unkown path type: %s' % path)

    def import_bboxes(self, file):
        data = io.load_data(file)
        assert data.shape[1] == 5
        self.bboxes = np.ndarray((self.images.shape[0], 4), dtype=np.float)
        self.bboxes[...] = float('nan')
        indices = self.find_images(data[:,0])
        self.bboxes[indices] = data[:,1:].astype(np.float32)

    def import_landmarks(self, file, dtype=float):
        data = io.load_data(file)
        self.landmarks = np.ndarray((self.images.shape[0], data.shape[1]-1), dtype=np.float)
        self.landmarks[...] = float('nan')
        indices = self.find_images(data[:,0])
        self.landmarks[indices] = data[:,1:].astype(np.float32)

    def import_features(self, file, features):
        data = io.load_data(file)
        self.features = np.ndarray((self.images.shape[0], features.shape[1]), dtype=np.float)
        self.features[...] = float('nan')
        indices = self.find_images(data[:,0])
        self.features[indices] =  features.astype(np.float32)
 
