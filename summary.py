import matplotlib.pyplot as plt
import numpy as np

def run(logdir):
    num_groups = 1
    num_inner = 1
    
    rank1 = []
    rank5 = []
    
    ha = 0
    hb = 0
    ma = 0
    mb = 0
    hc = 0
    mc = 0
    
    ranksets = []
    ta = 0
    fa = 0
    tr = 0
    fr = 0
    
    accScores = []
    rejScores = []
    
    for i in range(num_groups):
        for j in range(num_inner):
            rfile = open(logdir + '/result.txt')
            r1 = 0
            r5 = 0
    
            count = 0
            curve = [0.0 for k in range(0,5)]
    
            for line in rfile:
                if line == '':
                    continue
                rank = int(line.split(',')[1])
                score = float(line.split(',')[2])
    
                if rank >= 0:
                    rankA = max(rank-1,0)
                    for k in range(rankA,5):
                        curve[k] += 1
    
                count += 1
                if rank == 1 or rank == 0:
                    r1 += 1
    
                if rank <= 5 and rank >= 0:
                    r5 += 1
    
                '''if rank > 0:
                    ta += 1
                    accScores.append(score)
    
                if rank == 0:
                    tr += 1
                    rejScores.append(score)
    
                if rank == -1:
                    fr += 1
                    accScores.append(score)
    
                if rank == -2:
                    fa += 1
                    rejScores.append(score)'''
    
            for k in range(len(curve)):
                curve[k] /= count
    
    #        print(r1)
    #        print(count)
    
            ranksets.append(curve)
            rank1.append(r1 / float(count))
            rank5.append(r5 / float(count))
            print(rank1)
            print(rank5)
            return rank1, rank5
#meanr1 = sum(rank1)/len(rank1)
#meanr5 = sum(rank5)/len(rank5)
#
#print('Rank 1')
#print(' mean:  {0}'.format(meanr1))
#print(' median: {0}'.format(np.median(np.array(rank1))))
#print(' min:   {0}'.format(min(rank1)))
#print(' max:   {0}'.format(max(rank1)))
#print(' stdev: {0}'.format(np.std(np.array(rank1))))
#print
#if (fa+tr != 0):
#    print(' TA:   {0}'.format(ta/float(ta+fr)))
#    print(' FA:   {0}'.format(fa/float(fa+tr)))
#    print('{0},{1}\n'.format(ta/float(ta+fr),fa/float(fa+tr)))
#    plt.hist(np.array(rejScores),bins=5,normed=True,hold=True,color='r',alpha=0.5,label='No Match in Gallery')
#    plt.hist(np.array(accScores),bins=25,normed=True,hold=True,color='g',alpha=0.5,label='Match in Gallery')
#    plt.plot((-.83,-.83),(0,40),hold=True)
#    plt.xlabel('Match Score')
#    plt.ylabel('% of accept/reject images')
#    plt.legend()
#    plt.show()
#
#print('Rank 5')
#print(' mean:  {0}'.format(meanr5))
#print(' median: {0}'.format(np.median(np.array(rank5))))
#print(' min:   {0}'.format(min(rank5)))
#print(' max:   {0}'.format(max(rank5)))
#
#cmcs = np.array(ranksets) * 100
#minCMC = cmcs.min(0)
#maxCMC = cmcs.max(0)
#meanCMC = cmcs.mean(0)
#
#plt.plot(np.array(range(1,6)),minCMC,'r-',hold=True,label='min')
#plt.plot(np.array(range(1,6)),meanCMC,'b-',hold=True,label='mean')
#plt.plot(np.array(range(1,6)),maxCMC,'g-',hold=True,label='max')
#plt.axis([1,5,80,102])
#plt.yticks([i for i in range(80,101,2)])
#plt.xticks([1,2,3,4,5])
#plt.grid()
#plt.legend(loc='lower-right')
#plt.title('Closed-Set')
#plt.xlabel('Rank')
#plt.ylabel('Cumulative Accuracy (%)')
#plt.show()
