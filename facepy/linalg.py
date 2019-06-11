"""Linear Algebra Operations
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

# normalize one dimension of X as vectors.
def normalize(x, ord=None, axis=None, epsilon=10e-12):
	if axis is None:
		axis = len(x.shape) - 1
	norm = np.linalg.norm(x, ord=None, axis=axis, keepdims=True)
	x = x / (norm + epsilon)
	return x

def rc_indices(x, stack=True):
	r,c = x.shape
	# rows = np.repeat(np.arange(r)[:,None], c, axis=1)
	# cols = np.repeat(np.arange(c)[None,:], r, axis=0)
	rows, cols = np.meshgrid(np.arange(r), np.arange(c), indexing='ij')
	if stack:
		return np.stack([rows, cols], axis=2)
	else:
		return rows, cols