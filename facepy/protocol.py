"""Common protocols used for evaluation
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
import multiprocessing
import threading
import time

# compare a list of pairs of templates
# metricFunc is a function handle
# return: a vector of scores
def comparePairs(template_pairs, metricFunc, num_proc=8, log_info=False):
    proc_list = []
    result_array = multiprocessing.Array('f', len(template_pairs))
    print('# of pairs: %d' % len(template_pairs))
    def proc_job(pairs, start_idx, result_array):
        for i,pair in enumerate(pairs):
            if log_info and (i % len(pairs)//10) == 0:
                print('Comparing row: %d' % (start_idx+i))
            score = metricFunc(pair[0], pair[1])
            result_array[start_idx+i] = score

    split_size = len(template_pairs) // num_proc
    for i in range(num_proc):
        start_idx = i * split_size
        end_idx = len(template_pairs) if i==num_proc-1 else (i+1) * split_size
        p = multiprocessing.Process(target=proc_job, args=(template_pairs[start_idx:end_idx], start_idx, result_array))
        p.start()
        proc_list.append(p)
    for p in proc_list:
        p.join()

    scores = np.array(result_array)
    return scores

# compare every template in set1 to every template in set2
# set1 is of size m
# set2 is of size n
def compareSets(template_set1, template_set2, metricFunc, num_proc=8, log_info=False):
    m = len(template_set1)
    n = len(template_set2)
    print('Set1 size: %d Set2 size: %d' % (m,n))
    proc_list = []
    result_array = multiprocessing.Array('f', m*n)
    def proc_job(s1, s2, start_idx, n, result_array):
        for i,t1 in enumerate(s1):
            if log_info:
                print('Comparing row: %d' % (start_idx+i))
            for j,t2 in enumerate(s2):
                score = metricFunc(t1, t2)
                result_array[(start_idx+i)*n+j] = score

    split_size = len(template_set1) // num_proc
    for i in range(num_proc):
        start_idx = i * split_size
        end_idx = len(template_set1) if i==num_proc-1 else (i+1) * split_size
        p = multiprocessing.Process(target=proc_job, 
            args=(template_set1[start_idx:end_idx], template_set2, start_idx, n, result_array))
        p.start()
        proc_list.append(p)
    for p in proc_list:
        p.join()

    scores = np.array(result_array).reshape((m,n))
    return scores
