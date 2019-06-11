"""Plotting functions
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
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import scipy.misc as misc


def score_distribution(score_vec, label_vec, bins=100):
    score_pos = score_vec[label_vec]
    score_neg = score_vec[np.logical_not(label_vec)]
    h1 = plt.hist(score_pos, normed=True, alpha=0.7, edgecolor='black', bins=bins)
    h2 = plt.hist(score_neg, normed=True, alpha=0.7, edgecolor='black', bins=bins)
    # plt.legend([h1, h2], ['genuine', 'impostor'])
    plt.ylabel('frequency')
    plt.xlabel('similarity')
    plt.legend(['genuine','impostor'])
    plt.show()
    return

def show_image(img, width=4.0):    
    plt.cla()
    plt.imshow(img)    

    plt.axis('off')

    h, w = tuple(img.shape[0:2])
    plt.gcf().set_size_inches([width, 1.0 * width * h/w])
    plt.gca().set_position([0., 0., 1. ,1.])



def show_face(img, bboxes=None, landmarks=None, reorder_landmark=False, width=4.0):
    '''
        bboxes: [[x, y, w, h], ...]
        landmarks: see reorder_landmark
        reorder_landmark: default order is [x1 y1 x2 y2 ...], use reorder for [x1 x2 ... y1 y2 ...]
    '''
    if type(img) == str:
        img = plt.imread(img)

    show_image(img, width)

    if bboxes is not None:
        bboxes = np.array(bboxes)
        assert bboxes.ndim == 1 or bboxes.ndim == 2
        bboxes = bboxes.reshape([-1, 4])
        for i in range(bboxes.shape[0]):
            x, y, w, h = tuple(bboxes[i, :])
            plt.gca().add_patch(patches.Rectangle((x, y), w, h, fill=False,
                    edgecolor='r', linewidth=2))

    if landmarks is not None:
        landmarks = np.array(landmarks)
        assert landmarks.ndim == 1 or landmarks.ndim == 2
        if landmarks.ndim == 1:
            landmarks = landmarks[ None, :]
        for i in range(landmarks.shape[0]):
            if reorder_landmark:
                landmark = landmarks[i,:].reshape([2, -1]).transpose()
            else:
                landmark = landmarks[i,:].reshape([-1, 2])
            plt.plot(landmark[:,0], landmark[:,1], 'ro')

    plt.draw()


def show_face_list(list_file, delimiter=' ', annotation=None, reorder_landmark=False, width=4.0):
    '''
        reorder_landmark: default order is [x1 y1 x2 y2 ...], use reorder for [x1 x2 ... y1 y2 ...]
    '''
    with open(list_file, 'r') as f:
        lines = f.readlines()

    lines.sort()
    plt.ion()
    for line in lines:
        splits = line.strip().split(delimiter)
        img = splits[0]
        bbox = None
        landmark = None
        if annotation is None:
            pass
        elif annotation == 'bbox':
            bbox = map(float,splits[1:])
        elif annotation == 'landmark':
            landmark = map(float,splits[1:])
        else:
            raise ValueError('Unvalid annotation type: %s, it should be either \
                "bbox" or "landmark"')

        show_face(img, bbox, landmark, reorder_landmark, width)

        try:
            s = raw_input("Input 'q' to stop:") # wait for input from the user
        except:
            s = input("Input 'q' to stop:") # wait for input from the user
        if s == 'q':
            break


def show_embedding(features, images, full_size=45000, img_size=500):
    '''
        Visualize the embedding using images.
        args:
            features: N x 2 array
            images: a list of N images or an N x h x w x 3 array
            full_size: the size of the generated visualization image.
            img_size: the size of the images in the visualization.
        return:
            image: a image of full_size.
    '''

    # load embedding
    features = features - np.min(features, axis=0, keepdims=True)
    features = features / (1e-8 + np.max(features, axis=0, keepdims=True))
    n = features.shape[0]

    S = full_size # size of full embedding image
    image_full = np.zeros([S, S, 3], dtype=np.uint8)
    s = img_size # size of every single image

    for i in range(n):

        if i % 100==0:
            print('%d/%d...' % (i, n))

        # location
        a = int(np.ceil(features[i, 0] * (S-s)))
        b = int(np.ceil(features[i, 1] * (S-s)))
        a = a - (a % s)
        b = b - (b % s)

        if not image_full[a,b,0] == 0:
            continue

        if type(images[i]) is str:
            img = misc.imread(images[i], mode='RGB')
        else:
            img = images[i]

        img = misc.imresize(img, [s, s])

        image_full[a:a+s, b:b+s, :] = img;


    show_image(image_full)

    return image_full