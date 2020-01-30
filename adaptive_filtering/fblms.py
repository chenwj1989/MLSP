import numpy as np
import scipy.linalg as la
from numpy.fft import rfft, irfft

class FBLMS():
    '''
    Implementation of the fast block least mean squares algorithm (NLMS), 
    or frequency domain adaptive filter (FDAF)

    Parameters
    ----------
    length: int
        the length of the filter
    mu: float, optional
        the step size (default 0.01)
    L: int, optional
        block size (default is 1)
    nlms: bool, optional
        whether or not to normalize as in NLMS (default is False)
    '''
    
    def __init__(self, mu=0.01, B=1, nlms=False, constrained=True):
        
        self.nlms = nlms
        self.constrained = constrained

        # sketching parameters
        self.mu = mu
        self.B = B # block size
        
        self.reset()
        
    def reset(self):
        self.n = 0
        self.d = np.zeros((self.B))
        self.y = np.zeros((self.B))
        self.e = np.zeros((self.B))
        self.x = np.zeros((self.B * 2))
        self.W = np.zeros((self.B * 2), dtype=complex)
        #self.w = np.zeros((self.B))
        
    def update(self, x_n, d_n):
        '''
        Updates the adaptive filter with a new sample

        Parameters
        ----------
        x_n: float
            the new input sample
        d_n: float
            the new noisy reference signal
        '''
        
        # Update the internal buffers
        self.n += 1
        self.x[self.B+self.n] = x_n
        self.d[self.n] = d_n
        
        # Block update
        if self.n % self.B == 0:

            # block-update parameters
            X = rfft(self.x)
            y_2B = irfft( X * self.W)
            self.y = np.real(y_2B[self.B:])

            self.e = self.d - self.y
            E = rfft(np.concatenate([np.zeros(self.B), self.e])) # (2B)
            
            if self.nlms:
                norm = la.norm(X, axis=1)**2
                if self.B == 1:
                    X = X/norm[0]
                else:
                    X = (X.T/norm).T
                
            # Compute the correlation vector (gradient constraint)
            phi = np.einsum('i,i->i',X.conj(),E) # (2B)
            phi = irfft(phi)
            phi[self.B:] = 0
                   
            # Update the parameters of the filter
            self.W = self.W + self.mu*rfft(phi) 
            #self.w = irfft(self.W)[:B] 
            
            # sliding window
            self.n = 0
            self.x = np.concatenate(self.x[self.B:], np.zeros(self.B))
            self.d = np.zeros((self.B))
        
