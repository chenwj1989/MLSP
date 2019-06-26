# -*- coding:utf-8 -*-
import numpy as np
from scipy.stats import multivariate_normal

def scale_data(Y):
    #Y: [N*D]
    #N: number of data points
    #D: dimension of data
    for i in range(Y.shape[1]):
        max_ = Y[:, i].max()
        min_ = Y[:, i].min()
        Y[:, i] = (Y[:, i] - min_) / (max_ - min_)

    return Y

def gaussian(X, mu_k, cov_k):
    norm = multivariate_normal(mean=mu_k, cov=cov_k)
    return norm.pdf(X)

def gmm_em(X, K, iters):
    X = scale_data(X)
    N, D = X.shape
    #Init
    alpha = np.ones((K,1)) / K          #initially evenly distributed
    mu = np.random.rand(K, D)           #initially random mean
    cov = np.array([np.eye(D)] * K)     #intially diagonal covariance

    omega = np.zeros((N, K))

    for i in range(iters):

        #E-Step
        p = np.zeros((N, K))
        for k in range(K):
            p[:, k] = alpha[k] * gaussian(X, mu[k], cov[k])
        sumP = np.sum(p, axis=1)
        omega = p / sumP[:, None]

        #M-Step
        sumOmega = np.sum(omega, axis=0)  # [K]
        alpha = sumOmega / N              # alpha_k = sum(omega_k) / N
        for k in range(K):
            omegaX = np.multiply(X, omega[:, [k]])        # omega_k*X [N*D]
            mu[k] = np.sum(omegaX, axis=0) / sumOmega[k]  # mu[k]  = sum(omega_k*X) / sum(omega_k) : [D]

            X_mu_k = np.subtract(X, mu[k])                                         # (X - mu_k) : [N*D] - [D] = [N*D]
            omega_X_mu_k = np.multiply(omega[:, [k]], X_mu_k)                      # omega(X-mu_k) : [N*D]
            cov[k] = np.dot(np.transpose(omega_X_mu_k), X_mu_k) / sumOmega[k]      # sum(omega_i * (X_i-mu_k).T*(X_i-mu_k))  [D*D]

    return omega, alpha, mu, cov