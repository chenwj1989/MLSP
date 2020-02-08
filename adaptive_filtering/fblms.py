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
    constrained: bool, optional
        whether or not grdient constraint applies (default is True)
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
        self.xb = np.zeros((self.B))
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
        self.xb[self.n] = x_n
        self.d[self.n] = d_n
        self.n += 1
        
        # Block update
        if self.n % self.B == 0:
            self.process(self.xb, self.d)
            self.n =0


    def process(self, x_b, d_b, update=True): 
        '''
        Process a block data, and updates the adaptive filter (optional)

        Parameters
        ----------
        x_b: float
            the new input block signal
        d_b: float
            the new reference block signal
        update: bool, optional
            whether or not to update the filter coefficients
        '''  
        self.x = np.concatenate([self.x[self.B:], x_b])

        # block-update parameters
        X = fft(self.x)
        y_2B = ifft( X * self.W)
        y = np.real(y_2B[self.B:])

        e = d_b - y

        # Update the parameters of the filter
        if self.update:
            E = fft(np.concatenate([np.zeros(self.B), e])) # (2B)
            
            if self.nlms:
                norm = np.abs(X)**2 + 1e-10
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
                
            self.W = self.W + self.mu*phi
            self.w = np.real(ifft(self.W)[:self.B]) 
        
        return y, e

''' Alias : Fast LMS (FLMS) '''
FLMS = FBLMS

''' Alias : Frequency Domain Ddaptive Filter (FDAF) '''
FDAF = FBLMS
