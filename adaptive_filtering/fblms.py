import numpy as np
import scipy.linalg as la
from numpy.fft import fft, ifft

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
    B: int, optional
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
        self.x = np.zeros((self.B * 2))
        self.W = np.zeros((self.B * 2), dtype=complex)
        self.w = np.zeros((self.B))
        
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
        self.x[self.B+self.n] = x_n
        self.d[self.n] = d_n
        self.n += 1
        
        # Block update
        if self.n % self.B == 0:

            # block-update parameters
            X = fft(self.x)
            y_2B = ifft( X * self.W)
            y = y_2B[self.B:]

            e = self.d - y
            E = fft(np.concatenate([np.zeros(self.B), e])) # (2B)
            
            if self.nlms:
                norm = np.abs(X)**2
                E = E/norm
            # Set the upper bound of E, to prevent divergence
            m_errThreshold = 0.2
            Enorm = np.abs(E) # (2B)
            # print(E)
            for eidx in range(2*self.B):
                if Enorm[eidx]>m_errThreshold:
                    E[eidx] = m_errThreshold*E[eidx]/(Enorm[eidx]+1e-10) 

            # Compute the correlation vector (gradient constraint)
            phi = np.einsum('i,i->i',X.conj(),E) # (2B)
            phi = ifft(phi)
            phi[self.B:] = 0
            phi = fft(phi)
                   
            # Update the parameters of the filter
            self.W = self.W + self.mu*phi
            self.w = np.real(ifft(self.W)[:self.B]) 
            
            # sliding window
            self.n = 0
            self.x = np.concatenate([self.x[self.B:], np.zeros(self.B)])
            self.d = np.zeros((self.B))

        
