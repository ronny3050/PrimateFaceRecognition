"""Learning Algorithms
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
import sklearn
import h5py

class PCA:
	def __init__(self):
		self.mean = None
		self.eig_val = None
		self.eig_vec = None
		self.dims = None

	def fit(self, x, dims=None):
		self.mean = np.mean(x, axis=0, keepdims=True)
		cov = np.cov(x.T)
		self.eig_val, self.eig_vec = np.linalg.eig(cov)
		self.dims = dims if dims else x.shape[1]

	def transform(self, x, whiten=False):
		x = (x - self.mean) * eig_vec[:,:dims]
		if whiten:
			epsilon = 10e-8
			std = np.std(x, axis=0, keepdims=True)
			x = x / (std + epsilon)
		return x

	def save(self, filename):
		with h5py.File(filename, 'w') as f:
			f.create_dataset('mean', data=self.mean)
			f.create_dataset('eig_val', data=self.eig_val)
			f.create_dataset('eig_vec', data=self.eig_vec)
			f.create_dataset('dims', data=self.dims)

	def load(self, filename):
		with h5py.File(filename, 'r') as f:
			self.mean = np.array(f['mean'])
			self.eig_val = np.array(f['eig_val'])
			self.eig_vec = np.array(f['eig_vec'])
			self.dims = np.array(f['dims'])