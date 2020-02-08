
'''
Adaptive Filters Example
========================

In this example, we will run adaptive filters for system identification.
'''
from __future__ import division, print_function

import numpy as np
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt
import soundfile as sf
import pyroomacoustics as pra
from fblms import FBLMS
from pfblms import PFBLMS


# parameters
length = 2048        # the unknown filter length
SNR = 15           # signal to noise ratio

#Load far-end speech
farspeech, fs = sf.read("./samples/farspeech.wav")
n_samples = len(farspeech)

# the Room Impulse Response
# room dimension
room_dim = [5, 4, 6]
# Create the shoebox
room = pra.ShoeBox(
    room_dim,
    absorption=0.0,
    fs=fs,
    max_order=15,
    )
# source and mic locations
room.add_source([2, 2.5, 2], signal=farspeech)
room.add_microphone_array(
        pra.MicrophoneArray(
            np.array([[2, 2, 2]]).T, 
            room.fs)
        )
# run ism
room.simulate()
w = room.rir[0][0][:length]

# create a known driving signal
x = farspeech
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
        filter=pra.adaptive.BlockLMS(length, mu=1./2., L=2048, nlms=True), 
        error=np.zeros(n_samples),
        ),
    fblms=dict(
        filter=FBLMS(mu=1./32/2., B=2048, nlms=False), 
        error=np.zeros(n_samples),
        ),
    fblms_nlms=dict(
        filter=FBLMS(mu=1./4/2., B=2048, nlms=True), 
        error=np.zeros(n_samples),
        ),
    pfblms=dict(
        filter=PFBLMS(mu=1./2., B=64, M=32, nlms=True), 
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
