
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt

class OLA():
    '''
    Implementation of the overlap-add(OLA) convolution 


    Parameters
    ----------
    N: int
        the length of the filter
    P: int
        the length of one block
    K: int
        the length to do FFT, k > N + P - 1
    '''
    
    def __init__(self, N, P, K):

        self.N = N  # length of impulse response
        self.P = P  # length of segments
        self.K = K  # length of FFT size

        self.yk = np.zeros(K)
 
    def processBlock(self, xp, h):
        y_b = np.fft.irfft(np.fft.rfft(xp, self.K) * np.fft.rfft(h, self.K))
        self.yk[:self.K-self.P] = self.yk[self.P:] + y_b[:self.K-self.P]
        self.yk[self.K-self.P:] = y_b[self.K-self.P:]

        return self.yk[ : self.N+self.P-1 ]

    def reset(self):
        self.xk = np.zeros(self.K)
        self.yk = np.zeros(self.K)
