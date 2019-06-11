
from utils import *
import numpy as np


f = lambda t1, t2: metricFunc.avg_merge(metricFunc.euclidean(t1.data(),t2.data()))


dataset = dataset.Dataset(data=np.array([[1,0],[0,0],[0,1]]), image_list=['']*3)

template_pairs = dataset.get_template_pairs([(0,1), (1,2), (0,2)])
s1 = dataset.get_templates([0,1,2])

score = protocol.comparePairs(template_pairs, f)
scores = protocol.compareSets(s1, s1, f)
print score
print scores

label = np.array([False, True, False])
labels = np.array([	[False, False, False],
					[True, True, True],
					[True, False, False]])

TARs, FARs, thresholds = metrics.calcROC(scores.flatten(), labels.flatten())

print TARs
print FARs
print thresholds