
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import datetime
from ola import OLA
from ols import OLS


L = 10000000  # length of input signal
N = 2048  # length of impulse response
P = 2048  # length of segments
K = 4096


# generate input signal
x = sig.triang(L)
# generate impulse response
h = sig.triang(N)
# outpu 
y = np.zeros(L+P-1)

#direct convolution
start = datetime.datetime.now()
y = np.convolve(x, h, mode='full')
end = datetime.datetime.now()
print (end-start)

# overlap-add convolution
xp = np.zeros((L//P, P))
yp = np.zeros((L//P, N+P-1))
start = datetime.datetime.now()
for n in range(L//P):
    xp[n, :] = x[n*P:(n+1)*P]
    yp[n, :] = np.convolve(xp[n,:], h, mode='full')
    y[n*P:(n+1)*P+N-1] += yp[n, :]
y = y[0:N+L]
end = datetime.datetime.now()
print (end-start)

# overlap-add fast convolution
yp_fft = np.zeros((L//P, K))
y = np.zeros(L+P-1)
ola = OLA(N, P, K)
start = datetime.datetime.now()
for n in range(L//P):
    y[n*P:n*P+P+N-1] = ola.processBlock(x[n*P:(n+1)*P], h)
y = y[0:N+L]
end = datetime.datetime.now()
print (end-start)

# overlap-save fast convolution
ols = OLS(N, P, K)
start = datetime.datetime.now()
xx = np.zeros(L+K)
yy = np.zeros(L+K)
xx[:L] = x
for n in range((L+K)//P):
    yy[n*P:n*P+P] = ols.processBlock(xx[n*P:(n+1)*P], h)
y = yy[0:N+L]
end = datetime.datetime.now()
print (end-start)

