"""Affinity functions to compute scores
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


import numpy as np

def avgMerge(score_matrix):
    return score_matrix.mean().mean()

def maxMerge(score_matrix):
    return score_matrix.max().max()

def minMerge(score_matrix):
    return score_matrix.min().min()

# Compare between every row of x1 and every row of x2
def euclidean(x1,x2):
    assert x1.shape[1]==x2.shape[1]
    x2 = x2.transpose()
    x1_norm = np.sum(np.square(x1), axis=1, keepdims=True)
    x2_norm = np.sum(np.square(x2), axis=0, keepdims=True)
    dist = x1_norm + x2_norm - 2*np.dot(x1,x2)
    return dist

# Compare between every row of x1 and every row of x2
def cosineSimilarity(x1,x2):
    #assert x1.shape[1]==x2.shape[1]
    epsilon = 1e-10
    x2 = x2.transpose()
    x1_norm = np.sqrt(np.sum(np.square(x1), axis=1, keepdims=True))
    x2_norm = np.sqrt(np.sum(np.square(x2), axis=0, keepdims=True))
    x1 = x1 / (x1_norm+epsilon)
    x2 = x2 / (x2_norm+epsilon)
    dist = np.dot(x1, x2)
    return dist

# Compare between every row of x1 and x2
def euclidean_pair(x1, x2):
    assert x1.shape == x2.shape
    dist = np.sum(np.square(x1-x2), axis=1)
    return dist

# Compare between every row of x1 and x2
def cosine_pair(x1, x2):
    assert x1.shape == x2.shape
    epsilon = 1e-10
    x1_norm = np.sqrt(np.sum(np.square(x1), axis=1, keepdims=True))
    x2_norm = np.sqrt(np.sum(np.square(x2), axis=1, keepdims=True))
    x1 = x1 / (x1_norm+epsilon)
    x2 = x2 / (x2_norm+epsilon)
    dist = np.sum(x1 * x2, axis=1)
    return dist
