import numpy as np
import sys, os

cwd = os.getcwd()


N = 101000
L = 100

data_file = cwd + '/data'+str(L)

sigma = 1; mu = 0

np.random.seed(0)

dataF = np.random.rand(N,L)
dataI = np.zeros((N,L))
dataY = np.zeros((N,))
print dataY.shape

IdcLow = np.random.randint(0,L/2, size=N)
IdcHigh = np.random.randint(L/2,L, size=N)
for i in range(N):
    dataI[i,IdcLow[i]]=1.0
    dataI[i,IdcHigh[i]]=1.0
    dataY[i] = dataF[i,IdcLow[i]] + dataF[i,IdcHigh[i]]

data = np.zeros((N,2*L+1))
data[:,0] = dataY
data[:, 1::2] = dataF
data[:, 2::2] = dataI

try:
    os.remove(data_file)
except OSError:
    pass

np.savetxt(data_file, data, fmt='%.5f', delimiter = ',')
