import numpy 
# import scipy.io as sio
import h5py
# import hdf5storage
import os
import csv 
import numpy as np
import re

# def saveMatv7(fname, data, version=None):
# 	path = os.path.dirname(fname)
# 	name = os.path.basename(fname)
# 	hdf5storage.write(data, path, fname, matlab_compatible=True)

def load_data(filename, delimiter=r'[ ,\t]+'):
	with open(filename, 'r') as f:
		lines = f.readlines()
	lines = [re.split(delimiter, line.strip()) for line in lines]

	return np.array(lines, dtype=np.object)