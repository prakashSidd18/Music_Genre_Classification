import math
import numpy as np
data = np.loadtxt('mfcc.npy',delimiter=',',dtype=float)
print np.shape(data)
X = []
for sample in data:
    mfcc = np.reshape(sample, ((12,len(sample)/12)))
    cov = np.cov(mfcc)
    mean = np.mean(mfcc, axis=1)
    print np.shape(cov), np.shape(mean)
    cov = np.reshape(cov, (np.shape(cov)[0]*np.shape(cov)[1]))
    cov = np.concatenate((cov,mean),axis=0)
    X.append(cov)
np.savetxt('mfcc_gaussian.npy',X,delimiter=',')
