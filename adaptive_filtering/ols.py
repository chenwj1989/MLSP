
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt

class OLS():
    '''
    Implementation of the overlap-save(OLS) convolution 


    Parameters
    ----------
    N: int
        the length of the filter
    B: int
        the length of one input block
    K: int
        the length to do FFT, k > N + P - 1
    '''
    
    def __init__(self, N, B, K):
        self.N = N 
        self.B = B 
        self.K = K  

        self.xk = np.zeros(K)
        self.yk = np.zeros(K)
 
    def processBlock(self, xp, h):
        self.xk[:self.K-self.B] = self.xk[self.B:]
        self.xk[self.K-self.B:] = xp[:self.B]   #sliding window
        self.yk = np.fft.irfft(np.fft.rfft(self.xk, self.K) * np.fft.rfft(h, self.K))

        return self.yk[self.K-self.B:] #save the last block

    def reset(self):
        self.xk = np.zeros(self.K)
        self.yk = np.zeros(self.K)
