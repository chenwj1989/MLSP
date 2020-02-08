'''
aec
'''

from __future__ import print_function
import time
import numpy as np
import pyroomacoustics as pra
import matplotlib.pyplot as plt
import soundfile as sf
from fblms import FBLMS
from pfblms import PFBLMS


def genEcho(nearpath, farpath, micpath):
    #Load near-end speech
    nearspeech, fs = sf.read(nearpath)
    #Load far-end speech
    farspeech, fs = sf.read(farpath)

    plt.figure()
    plt.subplot(211)
    plt.plot(farspeech)
    plt.subplot(212)
    plt.plot(nearspeech)

    #Room Impulse Response
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
    room.add_source([2, 2.1, 2], signal=farspeech)
    room.add_source([2, 1.9, 2], signal=nearspeech)
    room.add_microphone_array(
            pra.MicrophoneArray(
                np.array([[2, 2, 2]]).T, 
                room.fs)
            )

    # run ism
    room.simulate()

    room.mic_array.to_wav(micpath, norm=True, bitdepth=np.int16)
    #farspeechecho = shoebox.mic_array.signals[0,:]
    return room.rir[0][0]

def aec(micspeechpath, farspeechpath, outputpath):
    #Load micorophone speech
    micspeech, fs = sf.read(micspeechpath)
    #Load far-end speech
    farspeech, fs = sf.read(farspeechpath)

    plt.figure()
    plt.subplot(311)
    plt.plot(farspeech)
    plt.subplot(312)
    plt.plot(micspeech)

    n_samples = len(farspeech)   # the number of samples to run
    outspeech = np.zeros(n_samples)

    # create a partioned fast block LMS filter 
    # parameters
    block = 64
    f = PFBLMS(mu=1./2., B=block, M=32, nlms=True)

    for _ in range(3):
        for i in range(int(n_samples/block)):
            index = np.arange( i*block, (i+1)*block, 1 )      
            _, e = f.process(farspeech[index], micspeech[index], update=True)
            outspeech[index] = e

    sf.write(outputpath, outspeech, fs)    
    plt.subplot(313)
    plt.plot(outspeech)
    plt.show()

#rir = genEcho("./samples/nearspeech.wav", "./samples/farspeech.wav", "micspeech.wav")
#plt.plot(rir)
aec( "./samples/micspeech.wav", "./samples/farspeech.wav", "outspeech.wav")
