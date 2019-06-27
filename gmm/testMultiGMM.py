

import matplotlib.pyplot as plt
from gmm_em import *


D = 2 # 2d data
K = 3 # 3 mixtures

#1:
n1 = 30
mu1 = [0, 1]
cov1 = [[0.3, 0],[0, 0.1]]
#2:
n2 = 70
mu2 = [2, 1]
cov2 = [[0.2, 0],[0, 0.3]]
#3:
n3 = 50
mu3 = [1.5, 3]
cov3 = [[0.3, 0],[0, 0.1]]

N = n1 + n2 + n3
alpha1 = n1/N
alpha2 = n2/N
alpha3 = n3/N


sample1 = np.random.multivariate_normal(mean=mu1, cov=cov1, size=n1)
sample2 = np.random.multivariate_normal(mean=mu2, cov=cov2, size=n2)
sample3 = np.random.multivariate_normal(mean=mu3, cov=cov3, size=n3)
Y = np.concatenate((sample1, sample2, sample3),axis=0)

plt.plot(sample1[:, 0], sample1[:, 1], "bo", label="class1")
plt.plot(sample2[:, 0], sample2[:, 1], "rs", label="class2")
plt.plot(sample3[:, 0], sample3[:, 1], "go", label="class3")
plt.legend(loc="best")
plt.title("Orginal GMM Data")
plt.show()

omega, alpha, mu, cov = gmm_em(Y, K, 100)

category = omega.argmax(axis=1).flatten().tolist()
class1 = np.array([Y[i] for i in range(N) if category[i] == 0])
class2 = np.array([Y[i] for i in range(N) if category[i] == 1])
class3 = np.array([Y[i] for i in range(N) if category[i] == 2])

plt.figure()
plt.plot(class1[:, 0], class1[:, 1], 'bo', label="class1")
plt.plot(class2[:, 0], class2[:, 1], 'rs', label="class2")
plt.plot(class3[:, 0], class3[:, 1], 'go', label="class3")
plt.legend(loc="best")
plt.title("GMM EM Output")
plt.show()

#Errors:
print("Original alpha: ", alpha1, alpha2, alpha3)
print("Estimated alpha: ", alpha[0], alpha[1], alpha[2])

print("Original mu: ", mu1, mu2, mu3)
print("Estimated mu: ", mu[0, :], mu[1,:], mu[2, :])

print("Original cov: ", cov1, cov2, cov3)
print("Estimated cov: ", cov[0, :, :], cov[1, : ,:], cov[2,:,:])