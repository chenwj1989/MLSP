
'''
Adaptive Filters Example
========================

In this example, we will run adaptive filters for system identification.
'''
from __future__ import division, print_function

import numpy as np
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt
import pyroomacoustics as pra
from fblms import FBLMS
from pfblms import PFBLMS


# parameters
length = 128        # the unknown filter length
n_samples = 10000   # the number of samples to run
SNR = 15           # signal to noise ratio

# the unknown filter (unit norm)
w = np.random.randn(length)
w /= np.linalg.norm(w)

# create a known driving signal
x = np.random.randn(n_samples)

# convolve with the unknown filter
d_clean = fftconvolve(x, w)[:n_samples]

# add some noise to the reference signal
d = d_clean + np.random.randn(n_samples) * 10**(-SNR / 20.)

# create a bunch adaptive filters
adfilt = dict(
    nlms=dict(
        filter=pra.adaptive.NLMS(length, mu=0.5), 
        error=np.zeros(n_samples),
        ),
    blocklms=dict(
        filter=pra.adaptive.BlockLMS(length, mu=1./128/2., L=128, nlms=False), 
        error=np.zeros(n_samples),
        ),
    fblms=dict(
        filter=FBLMS(mu=1./8/2., B=128, nlms=False), 
        error=np.zeros(n_samples),
        ),
    fblms_nlms=dict(
        filter=FBLMS(mu=1./2., B=128, nlms=True), 
        error=np.zeros(n_samples),
        ),
    pfblms=dict(
        filter=PFBLMS(mu=1./16/2., B=32, M=4, nlms=False), 
        error=np.zeros(n_samples),
        ),
    )

for i in range(n_samples):
    for algo in adfilt.values():
        algo['filter'].update(x[i], d[i])
        algo['error'][i] = np.linalg.norm(algo['filter'].w - w)

plt.plot(w)
for algo in adfilt.values():
    plt.plot(algo['filter'].w)
plt.title('Original and reconstructed filters')
plt.legend(['groundtruth'] + list(adfilt))

plt.figure()
for algo in adfilt.values():
    plt.semilogy(algo['error'])
plt.legend(adfilt)
plt.title('Convergence to unknown filter')
plt.show()
