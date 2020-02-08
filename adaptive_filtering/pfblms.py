import numpy as np
import scipy.linalg as la
from numpy.fft import fft, ifft

class PFBLMS():
    '''
    Implementation of the partitioned fast block least mean squares algorithm (PFBLMS), 
    or partitioned frequency domain adaptive filter (PFDAF),
    or partitioned frequency domain LMS (PFLMS),
    or multidelay block frequency adaptive filter (MDF)

    Parameters
    ----------
    length: int
        the length of the filter
    mu: float, optional
        the step size (default 0.01)
    B: int, optional
        block size (default is 1)
    M: int, optional
        number of partitions (default is 1)
    nlms: bool, optional
        whether or not to normalize as in NLMS (default is False)
    constrained: bool, optional
        whether or not grdient constraint applies (default is True)
    '''
    
    def __init__(self, mu=0.01, B=1, M=1,nlms=False, constrained=True):
        
        self.nlms = nlms
        self.constrained = constrained
        self.mu = mu
        self.B = B 
        self.M = M 
        self.reset()
        
    def reset(self):
        '''
        Clean data buffers, so the filter can be reused
        '''
        self.n = 0
        self.d = np.zeros((self.B))
        self.x = np.zeros((self.B * 2))
        self.xb = np.zeros((self.B))
        self.Xf = np.zeros((self.M, (self.B * 2)), dtype=complex)
        self.Wf = np.zeros((self.M, (self.B * 2)), dtype=complex)
        self.w = np.zeros((self.B*self.M))
      
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
        Xf_b = fft(self.x)

        # block-update parameters
        self.Xf[1:] = self.Xf[:self.M-1] # Xf : Mx2B  sliding window
        self.Xf[0] = Xf_b # Xf : Mx2B  sliding window
        y_2B = ifft(np.sum(self.Xf * self.Wf, axis=0)) # [Px2B] element multiply [Px2B] , then ifft
        y = np.real(y_2B[self.B:])

        e = d_b - y

        if update:
            E = fft(np.concatenate([np.zeros(self.B), e])) # (2B)

            if self.nlms:
                norm = np.abs(Xf_b)**2 + 1e-6
                E = E/norm

            # Set the upper bound of E, to prevent divergence
            m_errThreshold = 0.2
            Enorm = np.abs(E) # (2B)
            # print(E)
            for eidx in range(2*self.B):
                if Enorm[eidx]>m_errThreshold:
                    E[eidx] = m_errThreshold*E[eidx]/(Enorm[eidx]+1e-10)    

                
            # Update the parameters of the filter
            self.Wf = self.Wf + self.mu*E*self.Xf.conj()

            # Compute the correlation vector (gradient constraint)
            if self.constrained:
                waux = ifft(self.Wf, axis=1)
                waux[:, self.B:] = 0
                self.Wf = fft(waux, axis=1)

            self.w = np.real(ifft(self.Wf, axis=1)[:, :self.B].flatten())

        return y, e 